# Projects

Runnable experiments live here. The repository is intentionally structured so
that notebooks remain exploratory and the `projects/` folder contains the code
you can execute directly.

Files

- `model.py` - shared transformer blocks and helper functions.
- `train_tiny_gpt.py` - trains a small decoder-only language model.
- `generate_text.py` - generates samples from a saved checkpoint.
- `shakespeare_classifier.py` - character-level speaker classifier.

Examples

```bash
python projects/train_tiny_gpt.py --data-file input.txt --output-dir outputs
python projects/generate_text.py --checkpoint outputs/tiny_gpt.pth --prompt "One day"
python projects/shakespeare_classifier.py --input-file input.txt
```
