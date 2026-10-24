# Blind tagging sheet

100 queries (50 COCO, 50 TextCaps, one query per image, drawn with seed 47 and shuffled).
The source is not shown. Open `tag_sheet.html` to see each image beside its caption, then fill
the `tag` column of `tag_sheet.csv` (row ids `t001`..`t100`) with one of three labels.

- `visual`: the caption can be answered from what the picture shows (objects, people, scene,
  colours, actions). Reading any text in the image is not needed.
- `text`: the caption can only be answered by reading text in the image (a sign, label,
  number, name, title). Without the text, the caption would fit many images.
- `both`: the caption needs what is seen and what is written (for example a named object,
  or a scene plus the words on it).

Tag from the caption and image only. `tag_key.csv` maps the sheet ids back to real query ids
(which reveal the source); do not open it until all 100 rows are tagged.

The images live in the git-ignored `data/local/tagging_images/`; recreate them with
`uv run python scripts/fetch_gallery.py tagsheet` after `download`.
