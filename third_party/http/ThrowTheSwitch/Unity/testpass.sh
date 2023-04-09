# Navigate into target directory.
cd $/Users/uwe/Documents/Programmierung/C/03_Projects/03_DoubleMatrix/bazel-testlogs/tests

# Required extension for unity_to_junit python scripts.
test_pass_ext="testpass"
test_fail_ext="testfail"

# Reset results files
> results.${test_pass_ext}
> results.${test_fail_ext}

# Move test result files into test result directory
mv ${test_object_dir}/*.txt .

# Find PASS and FAIL from Unity test results, saving them
# into their own files.
for file in *.txt; do
    grep "PASS" ${file} >> results.${test_pass_ext}
    grep -e "FAIL" -e "IGNORE" ${file} >> results.${test_fail_ext}
done

# Run unity_to_junit parsing tool.
python ${unity_dir}/auto/unity_to_junit.py