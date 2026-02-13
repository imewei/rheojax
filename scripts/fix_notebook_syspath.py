"""Fix sys.path.insert pattern in all example notebooks.

Replaces the incorrect:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), "examples"))
With the correct:
    sys.path.insert(0, os.path.dirname(os.path.abspath("")))

Notebooks run from examples/<family>/, so dirname(abspath("")) = examples/.
"""

import glob

old = 'os.path.join(os.path.dirname(os.path.abspath("")), "examples")'
new = 'os.path.dirname(os.path.abspath(""))'

# In JSON, quotes are escaped
old_json = old.replace('"', '\\"')
new_json = new.replace('"', '\\"')

fixed = 0
checked = 0

for f in sorted(glob.glob("examples/**/*.ipynb", recursive=True)):
    checked += 1
    with open(f) as fh:
        content = fh.read()

    if old_json in content:
        content = content.replace(old_json, new_json)
        with open(f, "w") as fh:
            fh.write(content)
        fixed += 1
        print(f"  Fixed: {f}")

print(f"\nChecked {checked} notebooks, fixed {fixed}")
