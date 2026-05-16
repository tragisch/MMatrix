#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

FILE_MAP = {
    "st.h": "st.md",
    "st_batchnorm.h": "st_batchnorm.md",
    "st_conv.h": "st_conv.md",
    "st_pool.h": "st_pool.md",
    "st_shape_ops.h": "st_shape_ops.md",
}

TITLE_MAP = {
    "st.h": "Public Core Tensor API",
    "st_batchnorm.h": "BatchNorm API",
    "st_conv.h": "Convolution API",
    "st_pool.h": "Pooling API",
    "st_shape_ops.h": "Shape/View Operations API",
}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def collect_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return norm_text("".join(elem.itertext()))


def format_brief(member: ET.Element) -> str:
    brief = collect_text(member.find("briefdescription"))
    if brief:
        return brief
    detail = collect_text(member.find("detaileddescription"))
    return detail


def parse_param_docs(member: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    detail = member.find("detaileddescription")
    if detail is None:
        return out
    for plist in detail.findall(".//parameterlist[@kind='param']"):
        for pitem in plist.findall("parameteritem"):
            names = [norm_text("".join(n.itertext())) for n in pitem.findall("./parameternamelist/parametername")]
            pdesc = norm_text("".join(pitem.findtext("parameterdescription", default="") or ""))
            if not pdesc:
                pdesc = collect_text(pitem.find("parameterdescription"))
            for n in names:
                if n:
                    out[n] = pdesc
    return out


def parse_return_doc(member: ET.Element) -> str:
    detail = member.find("detaileddescription")
    if detail is None:
        return ""
    for sect in detail.findall(".//simplesect[@kind='return']"):
        txt = collect_text(sect)
        if txt:
            return txt
    return ""


def signature(member: ET.Element) -> str:
    rtype = norm_text("".join((member.findtext("type") or "").split()))
    name = norm_text(member.findtext("name") or "")
    args = norm_text(member.findtext("argsstring") or "()")
    if rtype:
        return f"{rtype} {name}{args}"
    return f"{name}{args}"


def render_enum(member: ET.Element) -> str:
    lines = [f"### `{member.findtext('name', default='(anonymous)')}`", ""]
    brief = format_brief(member)
    if brief:
        lines.append(brief)
        lines.append("")
    values = member.findall("enumvalue")
    if values:
        lines.append("Values:")
        lines.append("")
        for v in values:
            vname = v.findtext("name", default="")
            vbrief = format_brief(v)
            if vbrief:
                lines.append(f"- `{vname}` — {vbrief}")
            else:
                lines.append(f"- `{vname}`")
        lines.append("")
    return "\n".join(lines)


def render_typedef(member: ET.Element) -> str:
    name = member.findtext("name", default="")
    tpe = norm_text(member.findtext("type") or "")
    brief = format_brief(member)
    lines = [f"- `{tpe} {name}`" if tpe else f"- `{name}`"]
    if brief:
        lines[-1] += f" — {brief}"
    return "\n".join(lines)


def render_function(member: ET.Element) -> str:
    name = member.findtext("name", default="")
    lines = [f"### `{name}`", "", f"`{signature(member)}`", ""]

    brief = format_brief(member)
    if brief:
        lines.append(brief)
        lines.append("")

    params = member.findall("param")
    pdocs = parse_param_docs(member)
    if params:
        lines.append("Parameters:")
        lines.append("")
        for p in params:
            ptype = norm_text(collect_text(p.find("type")))
            pname = norm_text(collect_text(p.find("declname")))
            if not pname:
                pname = "(unnamed)"
            desc = pdocs.get(pname, "")
            item = f"- `{pname}` (`{ptype}`)" if ptype else f"- `{pname}`"
            if desc:
                item += f": {desc}"
            lines.append(item)
        lines.append("")

    ret = parse_return_doc(member)
    if ret:
        lines.append(f"Returns: {ret}")
        lines.append("")

    return "\n".join(lines)


def build_markdown(file_name: str, xml_file: Path) -> str:
    root = ET.parse(xml_file).getroot()
    cdef = root.find("compounddef")
    if cdef is None:
        raise ValueError(f"compounddef missing in {xml_file}")

    title = TITLE_MAP.get(file_name, "API")
    brief = collect_text(cdef.find("briefdescription"))

    enums: list[ET.Element] = []
    typedefs: list[ET.Element] = []
    funcs: list[ET.Element] = []

    for member in cdef.findall(".//memberdef"):
        kind = member.attrib.get("kind", "")
        if kind == "enum":
            enums.append(member)
        elif kind == "typedef":
            typedefs.append(member)
        elif kind == "function":
            funcs.append(member)

    out: list[str] = []
    out.append(f"# `{file_name}` – {title}")
    out.append("")
    if brief:
        out.append(brief)
        out.append("")

    if enums or typedefs:
        out.append("## Types")
        out.append("")
        if enums:
            for e in enums:
                out.append(render_enum(e))
        if typedefs:
            out.append("### Typedefs")
            out.append("")
            for td in typedefs:
                out.append(render_typedef(td))
            out.append("")

    if funcs:
        out.append("## Functions")
        out.append("")
        for f in funcs:
            out.append(render_function(f))

    return textwrap.dedent("\n".join(out)).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Doxygen XML file docs to Markdown pages.")
    ap.add_argument("--xml-dir", required=True, help="Path to Doxygen xml output directory")
    ap.add_argument("--output-dir", required=True, help="Markdown output directory")
    args = ap.parse_args()

    xml_dir = Path(args.xml_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = xml_dir / "index.xml"
    if not index.exists():
        raise FileNotFoundError(f"index.xml not found: {index}")

    root = ET.parse(index).getroot()

    file_compounds = [c for c in root.findall("compound") if c.attrib.get("kind") == "file"]
    ref_by_name: dict[str, str] = {}
    for c in file_compounds:
        name = norm_text(c.findtext("name") or "")
        refid = c.attrib.get("refid", "")
        if name and refid:
            ref_by_name[name] = refid

    generated = 0
    for header, out_name in FILE_MAP.items():
        refid = ref_by_name.get(header)
        if not refid:
            print(f"[warn] No Doxygen compound found for {header}; skipping")
            continue
        xml_file = xml_dir / f"{refid}.xml"
        if not xml_file.exists():
            print(f"[warn] Missing XML file for {header}: {xml_file}")
            continue

        md = build_markdown(header, xml_file)
        (out_dir / out_name).write_text(md, encoding="utf-8")
        generated += 1
        print(f"[ok] {header} -> {out_name}")

    if generated == 0:
        raise RuntimeError("No API pages generated; check Doxygen INPUT/FILE_PATTERNS.")

    print(f"Generated {generated} API Markdown files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
