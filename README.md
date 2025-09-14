# 🌱 Plant Disease Detection System

![Plant Disease Detection](image.png)

A comprehensive web application for detecting plant diseases using deep learning. This system uses a Convolutional Neural Network (CNN) to identify diseases in plant leaves from uploaded images, supporting 38 different disease classes across multiple plant types.

## 🚀 Features

- **Real-time Disease Detection**: Upload plant leaf images for instant disease analysis
- **38 Disease Classes**: Supports detection across multiple plant types including Apple, Tomato, Potato, Corn, and more
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience
- **Detailed Disease Information**: Provides symptoms, descriptions, and treatment recommendations
- **Analytics Dashboard**: Track detection history and view comprehensive statistics
- **Confidence Scoring**: Shows prediction confidence levels with visual charts
- **Disease Database**: Searchable database of all supported diseases
- **Model Information**: Detailed insights into the CNN architecture and dataset

## 🎯 Supported Plant Types & Diseases

### Plant Types (14 types)
- Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### Disease Classes (38 total)
- **Healthy conditions** for all plant types
- **Common diseases** including:
  - Bacterial spot, Black rot, Early blight, Late blight
  - Powdery mildew, Cedar apple rust, Leaf blight
  - Viral infections (Yellow Leaf Curl Virus, Mosaic virus)
  - And many more...

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Dependencies
```bash
pip install streamlit tensorflow numpy pillow pandas plotly
```

### Project Structure
```
plant-disease-detection/
├── app.py                      # Main Streamlit application
├── plant_disease_model.keras   # Trained CNN model
├── image.png                   # Application screenshot
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

### Running the Application
1. Clone or download this repository
2. Ensure the trained model file (`plant_disease_model.keras`) is in the same directory
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
5. Open your browser and navigate to `http://localhost:8501`

## 📊 Dataset Information

This project uses the **New Plant Diseases Dataset** from Kaggle:
- **Source**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Size**: ~87,000 RGB images
- **Classes**: 38 different disease classes
- **Split**: 80/20 training/validation ratio
- **Augmentation**: Offline augmentation applied for better generalization

### Dataset Characteristics
- **Image Format**: RGB color images
- **Input Size**: 128×128 pixels (resized during preprocessing)
- **Categories**: Healthy and diseased crop leaves
- **Quality**: High-resolution images with clear disease symptoms

## 🧠 Model Architecture

The disease detection model is built using a Convolutional Neural Network (CNN) trained with TensorFlow/Keras:

- **Training Notebook**: [Plant Disease Detection CNN](https://www.kaggle.com/code/virajinduruwa/plant-disease-detection-cnn)
- **Input Shape**: 128×128×3 (RGB images)
- **Architecture**: Deep CNN with multiple convolutional and pooling layers
- **Output**: 38 classes (softmax activation)
- **Framework**: TensorFlow/Keras

### Model Performance
- High accuracy on validation set
- Robust to various lighting conditions
- Effective across different plant types
- Real-time inference capabilities

## 💻 Application Features

### 🔍 Disease Detection Page
- **Image Upload**: Support for JPG, PNG, BMP, TIFF formats
- **Real-time Analysis**: Instant disease detection results
- **Confidence Scoring**: Visual confidence charts for predictions
- **Disease Information**: Detailed symptoms and treatment recommendations
- **Image Preprocessing**: Automatic image optimization for best results

### 📊 Model Information Page
- **Architecture Details**: Complete model summary and parameters
- **Dataset Statistics**: Distribution of plant types and disease classes
- **Visual Analytics**: Charts showing dataset composition

### 🌿 Disease Database Page
- **Searchable Database**: Find specific diseases or plant types
- **Comprehensive Information**: Symptoms, descriptions, and treatments
- **Organized Display**: Grouped by plant type with expandable sections

### 📈 Analytics Dashboard
- **Detection History**: Track all previous analyses
- **Statistical Insights**: Summary metrics and trends
- **Visual Charts**: Distribution of plant types and health status
- **Data Export**: Download history as CSV files

## 🎨 User Interface

The application features a modern, responsive design with:
- **Clean Layout**: Intuitive navigation and organization
- **Color-coded Results**: Green for healthy, red for diseased plants
- **Interactive Charts**: Plotly-powered visualizations
- **Mobile Friendly**: Responsive design for various screen sizes
- **Professional Styling**: Custom CSS for enhanced user experience

## 🔧 Technical Implementation

### Key Technologies
- **Frontend**: Streamlit for web interface
- **Backend**: Python with TensorFlow for model inference
- **Visualization**: Plotly for interactive charts
- **Image Processing**: PIL (Pillow) for image handling
- **Data Management**: Pandas for data manipulation

### Model Integration
- **Caching**: Efficient model loading with Streamlit caching
- **Preprocessing Pipeline**: Automated image preprocessing
- **Batch Processing**: Optimized for single and multiple predictions
- **Error Handling**: Comprehensive error management

## 📝 Usage Tips

### For Best Results
- **Image Quality**: Use clear, well-lit images
- **Focus Area**: Ensure the affected leaf area is prominent
- **Image Clarity**: Avoid blurry or dark images
- **Composition**: Leaf should fill most of the frame
- **Background**: Minimize background distractions

### Supported Scenarios
- **Field Diagnosis**: Use in agricultural settings
- **Educational Purposes**: Learn about plant diseases
- **Research Applications**: Analyze plant health patterns
- **Preventive Care**: Early disease detection

## 🔮 Future Enhancements

- **Mobile App**: Native mobile application development
- **Batch Processing**: Multiple image analysis
- **Treatment Tracking**: Follow-up and treatment monitoring
- **Expert System**: Integration with agricultural expertise
- **Multilingual Support**: Interface in multiple languages
- **API Development**: REST API for third-party integrations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional disease classes
- Model performance optimization
- UI/UX enhancements
- Documentation improvements
- Testing and validation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Dataset**: Thanks to the creators of the New Plant Diseases Dataset on Kaggle
- **Model Training**: Based on the CNN implementation by virajinduruwa
- **Libraries**: TensorFlow, Streamlit, Plotly, and other open-source libraries
- **Community**: Agricultural and machine learning communities for inspiration

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in this repository
- Check the troubleshooting section in the application
- Review the model information page for technical details

---

**Made with 🌱 for sustainable agriculture and plant health monitoring**
