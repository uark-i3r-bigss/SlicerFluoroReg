# Sourced by the scripts that run 3D Slicer, after setting repo. Sets module_names to the modules added with
# add_subdirectory(<module>) in the top-level CMakeLists.txt (same rule as scripts/package_extension.py),
# module_dirs to their directories and module_pythonpath to those directories joined with ':'.

mapfile -t module_names < <(sed 's/\(^\|[[:space:]]\)#.*$//' "$repo/CMakeLists.txt" |
    grep -o '\badd_subdirectory([[:space:]]*[^[:space:])]*[[:space:]]*)' |
    sed 's/^add_subdirectory([[:space:]]*//; s/[[:space:]]*)$//')
if [ ${#module_names[@]} -eq 0 ]; then
    echo "no add_subdirectory(<module>) in $repo/CMakeLists.txt" >&2
    exit 2
fi
module_dirs=()
for _name in "${module_names[@]}"; do
    module_dirs+=("$repo/$_name")
done
unset _name
module_pythonpath="$(IFS=:; echo "${module_dirs[*]}")"
