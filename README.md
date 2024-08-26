# Interactive Images

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/python-%3E%3D3.11-%233776AB?logo=python&logoColor=%234584b6&labelColor=%23646464)

While browsing on LinkedIn I came across [a demo from Cinemersive Labs](https://www.linkedin.com/feed/update/urn:li:activity:7209214244821577731/) showing 3D six-degrees-of-freedom photographs created from single photo shots.
Wanting to make an intro towards generative 3D, this repo will attempt to follow a similar idea to create interactive scenes from a single (or more) image.

## Quickstart
The project includes docker and docker-compose files for easy setup and a .devcontainer configuration folder for use in VSCode that also auto-installs extensions and sets up workspace settings.

## Development (recommendation)
I recommend using Visual Studio Code (VS Code) with the Remote Development extension to work on this project. This allows you to open the project as a container and have all dependencies automatically set up.

To use VS Code, follow these steps:

1. Open VS Code.
2. Install the Remote Development extension if you haven't already.
3. Press Ctrl/Cmd + Shift + P and select "Dev Containers: Open Folder in Container..." from the command palette.
4. Wait for the container to set up and start development.

## TODO
- [x] Find out why the outputs from of [ZoeDepth](https://github.com/isl-org/ZoeDepth) from the original repo are different that the one from the [🤗 implementation](https://huggingface.co/docs/transformers/v4.43.4/en/model_doc/zoedepth) (+fix). Track the updates in [this issue on 🤗 Transformers](https://github.com/huggingface/transformers/issues/32381)
- [x] Begin by cloning the "Image to 3D" tab functionality of the [ZoeDepth 🤗 demo](https://huggingface.co/spaces/shariqfarooq/ZoeDepth)
- [x] Understand "intrinsic and extrinsic camera parameters" and how to use them in 3D a bit better
- [x] Render images from 3D mesh
- [x] Use [Pytorch3D](https://pytorch3d.org/) whenever possible
- [ ] Finish cleaning up `Image_to_3D` notebook and complete transition to scripts for main functionality
- [ ] Fill missing parts of the image using Stable Diffusion or other similar generative model
- [ ] Probably also use a depth control net for the generated images
- [ ] Could/should we use a video generation model instead?
