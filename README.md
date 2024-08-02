# Markeplace User Review Image Recognition Service

This project provides a service for recognizing images in the product review section of a marketplace.

## Applications

The service consists of three applications that can classify images:
- **clip_cb**: Uses the pre-trained CLIP model as an encoder and CatBoost for classification.
- **easyocr_cb**: Uses text recognition with EasyOCR, TfidfVectorizer for text vectorization, and CatBoost for classification.
- **metamodel**: Based on blending of two previous models.

## Requirements

- Docker Compose

## Installation

Clone the repository and navigate to the project root directory:

```bash
git clone https://github.com/egoRRyabinin/wbuserrep_service.git
cd wbuserrep_service
# Build and run the Docker containers:
docker compose build
docker compose up -d
# Wait for the models to load.
```
## Usage
- Open your browser and go to http://127.0.0.1:5000.
- Upload the necessary images.
- Select the microservice to get the prediction.

## Example
![Example Image](https://github.com/egoRRyabinin/wbuserrep_service/blob/master/images/example.png)
