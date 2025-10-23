#!/bin/bash
# Emotion Dataset Download Script
# Downloads RAVDESS, CREMA-D, MELD, and other emotion datasets

echo "📥 Downloading Emotion Datasets"
echo "==============================="

cd /workspace
mkdir -p emotion_datasets

# Function to download and verify datasets
download_dataset() {
    local name=$1
    local url=$2
    local dir=$3
    
    echo ""
    echo "📦 Downloading $name..."
    if [ -d "emotion_datasets/$dir" ]; then
        echo "   ✅ $name already exists, skipping..."
        return 0
    fi
    
    mkdir -p "emotion_datasets/$dir"
    cd "emotion_datasets/$dir"
    
    wget -q --show-progress "$url" -O dataset.zip 2>&1
    if [ $? -eq 0 ]; then
        unzip -q dataset.zip
        rm dataset.zip
        echo "   ✅ $name downloaded successfully"
        cd /workspace
        return 0
    else
        echo "   ⚠️  Direct download failed, trying alternative method..."
        cd /workspace
        return 1
    fi
}

# Dataset 1: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
echo ""
echo "📦 Dataset 1: RAVDESS"
echo "   Source: Zenodo"
echo "   Emotions: 8 emotions × 24 actors"
echo "   Size: ~2 hours of speech"
echo ""
if [ ! -d "emotion_datasets/ravdess" ]; then
    mkdir -p emotion_datasets/ravdess
    cd emotion_datasets/ravdess
    
    # RAVDESS is freely available from Zenodo
    echo "   Please download RAVDESS manually from:"
    echo "   https://zenodo.org/record/1188976"
    echo "   Extract to: /workspace/emotion_datasets/ravdess/"
    echo ""
    echo "   Or use:"
    echo "   wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    cd /workspace
else
    echo "   ✅ RAVDESS already exists"
fi

# Dataset 2: CREMA-D (Crowd Sourced Emotional Multimodal Actors Dataset)
echo ""
echo "📦 Dataset 2: CREMA-D"
echo "   Source: GitHub"
echo "   Emotions: 6 emotions × 91 actors"
echo "   Size: ~7 hours of speech"
echo ""
if [ ! -d "emotion_datasets/crema_d" ]; then
    mkdir -p emotion_datasets/crema_d
    cd emotion_datasets/crema_d
    
    echo "   Please download CREMA-D from:"
    echo "   https://github.com/CheyneyComputerScience/CREMA-D"
    echo "   Extract to: /workspace/emotion_datasets/crema_d/"
    
    cd /workspace
else
    echo "   ✅ CREMA-D already exists"
fi

# Dataset 3: MELD (Multimodal EmotionLines Dataset)
echo ""
echo "📦 Dataset 3: MELD"
echo "   Source: GitHub"
echo "   Emotions: Conversational emotions"
echo "   Size: ~13 hours of speech"
echo ""
if [ ! -d "emotion_datasets/meld" ]; then
    mkdir -p emotion_datasets/meld
    cd emotion_datasets/meld
    
    echo "   Downloading MELD from GitHub..."
    git clone https://github.com/declare-lab/MELD emotion_datasets/meld
    
    if [ $? -eq 0 ]; then
        echo "   ✅ MELD downloaded successfully"
    else
        echo "   ⚠️  Please clone manually:"
        echo "   git clone https://github.com/declare-lab/MELD"
    fi
    
    cd /workspace
else
    echo "   ✅ MELD already exists"
fi

# Dataset 4: EmoDB (Berlin Database of Emotional Speech) - Small but high quality
echo ""
echo "📦 Dataset 4: EmoDB"
echo "   Source: EMO-DB"
echo "   Emotions: 7 emotions × 10 speakers"
echo "   Size: ~1 hour of professional speech"
echo ""
if [ ! -d "emotion_datasets/emodb" ]; then
    mkdir -p emotion_datasets/emodb
    cd emotion_datasets/emodb
    
    echo "   Please download EmoDB from:"
    echo "   http://emodb.bilderbar.info/download/download.zip"
    echo "   Extract to: /workspace/emotion_datasets/emodb/"
    
    cd /workspace
else
    echo "   ✅ EmoDB already exists"
fi

# Create dataset info file
cat > emotion_datasets/README.md << 'EOF'
# Emotion Datasets for Oviya

## Downloaded Datasets

### 1. RAVDESS
- **Emotions**: neutral, calm, happy, sad, angry, fearful, disgusted, surprised
- **Actors**: 24 (12 male, 12 female)
- **Size**: ~2 hours
- **Quality**: Clean, acted speech
- **License**: CC BY-NC-SA 4.0

### 2. CREMA-D
- **Emotions**: angry, disgusted, fearful, happy, neutral, sad
- **Actors**: 91 (48 male, 43 female)
- **Size**: ~7 hours
- **Quality**: Natural prosody
- **License**: OpenDataCommons

### 3. MELD
- **Emotions**: joy, sadness, anger, fear, disgust, surprise, neutral
- **Context**: TV show dialogues
- **Size**: ~13 hours
- **Quality**: Conversational, empathetic
- **License**: MIT

### 4. EmoDB
- **Emotions**: anger, boredom, anxiety, happiness, sadness, disgust, neutral
- **Speakers**: 10 professional actors
- **Size**: ~1 hour
- **Quality**: Studio quality, German accent
- **License**: Free for research

## Usage

Each dataset provides unique emotional expressions:
- **RAVDESS**: Clean, consistent acted emotions
- **CREMA-D**: Natural prosody variations
- **MELD**: Conversational empathy and nuance
- **EmoDB**: Professional, studio-quality recordings

## Extraction

Use `extract_dataset_emotions.py` to extract samples and create embeddings.
EOF

echo ""
echo "📋 Dataset download instructions created"
echo "📁 Location: /workspace/emotion_datasets/"
echo ""
echo "🎯 Next Steps:"
echo "   1. Download datasets manually from the provided links"
echo "   2. Extract them to /workspace/emotion_datasets/[dataset_name]/"
echo "   3. Run extract_dataset_emotions.py to process them"


