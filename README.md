# Z-Image.swift

Swift port of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) using [mlx-swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon.

## Requirements

- macOS 14.0+
- Apple Silicon
- Swift 5.9+

## Installation

### Download Precompiled Binary

Grab the latest signed ZImageCLI binary from the [releases page](https://github.com/mzbac/zimage.swift/releases). The asset is shipped as a zipped bundle:

```bash
curl -L https://github.com/mzbac/zimage.swift/releases/latest/download/zimage.macos.arm64.zip \
  -o zimage.macos.arm64.zip
unzip -o zimage.macos.arm64.zip -d z-image-cli
cd z-image-cli
chmod +x ZImageCLI
./ZImageCLI -h
```

### Building from Source

```bash
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

The CLI binary will be available at `.build/xcode/Build/Products/Release/ZImageCLI`.

## Usage

```bash
ZImageCLI -p "A beautiful mountain landscape at sunset" -o output.png
```

For all available options:

```bash
ZImageCLI -h
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --prompt` | Text prompt (required) | - |
| `--negative-prompt` | Negative prompt | - |
| `-W, --width` | Output width | 1024 |
| `-H, --height` | Output height | 1024 |
| `-s, --steps` | Inference steps | 9 |
| `-g, --guidance` | Guidance scale | 3.0 |
| `--seed` | Random seed | random |
| `-o, --output` | Output path | z-image.png |
| `-m, --model` | Model path or HuggingFace ID | Tongyi-MAI/Z-Image-Turbo |
| `--cache-limit` | GPU memory cache limit in MB | unlimited |

## Examples

```bash
# Basic generation
ZImageCLI -p "a cute cat sitting on a windowsill" -o cat.png

# Portrait image with custom size
ZImageCLI -p "portrait of a woman in renaissance style" -W 768 -H 1152 -o portrait.png

# Using quantized model for lower memory usage
ZImageCLI -p "a futuristic city at night" -m mzbac/Z-Image-Turbo-8bit -o city.png

# With memory limit
ZImageCLI -p "abstract art" --cache-limit 2048 -o art.png
```

## Example Output

| Prompt | Output |
|--------|--------|
| A dramatic, cinematic japanese-action scene in a edo era Kyoto city. A woman named Harley Quinn from the movie "Birds of Prey" in colorful, punk-inspired comic-villain attire walks confidently while holding the arm of a serious-looking man named John Wick played by Keanu Reeves from the fantastic film John Wick 2 in a black suit, her t-shirt says "Birds of Prey", the characters are capture in a postcard held by a hand in front of a beautiful realistic city at sunset and there is cursive writing that says "ZImage, Now in MLX" | ![Output](examples/z-image.png) |

## Quantization

Quantize the model to reduce memory usage:

```bash
ZImageCLI quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --bits 8 --group-size 32 --verbose
```

### Performance

| Model | Memory | Time (1024x1024) |
|-------|--------|------------------|
| BF16 | ~21 GB | ~46s |
| 8-bit quantized | ~7.5 GB | ~44s |

*Tested on Apple M2 Ultra*

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework for Apple Silicon
- [swift-transformers](https://github.com/huggingface/swift-transformers) - Tokenizer support
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) - CLI argument parsing

## License

MIT License
