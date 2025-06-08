# Setzt voraus, dass compile_commands.json vorhanden ist
clang-tidy file:
    clang-tidy {{ file }}

# Alle *.c Dateien im src/ Verzeichnis prüfen
clang-tidy-all:
    find src -name '*.c' | xargs -n 1 clang-tidy --quiet -p=.

# Beispiel: Nur bestimmte Checks aktivieren
clang-tidy-strict file:
    clang-tidy {{ file }} --quiet -p=. \
      -checks='clang-analyzer-*,cppcoreguidelines-*,bugprone-*,performance-*,readability-*'

# Nur dry-run (zeigt an, was geprüft werden würde)
dry-run:
    echo "Würde prüfen: $(find src -name '*.c')"
