import re

filepath = "env/Lib/site-packages/torchvision/transforms/functional.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

new_content, _ = re.subn(
    r'from\s+PIL\s+import\s+([^#\n]*?)\s*,\s*PILLOW_VERSION',
    r'from PIL import \1',
    content
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)