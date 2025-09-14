import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .result-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .healthy-result {
        border-left: 5px solid #4CAF50;
        background: #e8f5e8;
    }
    
    .diseased-result {
        border-left: 5px solid #f44336;
        background: #ffebee;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .sidebar-content {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Class names for plant diseases
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease information dictionary
DISEASE_INFO = {
    'Apple_scab': {
        'description': 'A fungal disease causing dark, scabby lesions on leaves and fruit.',
        'symptoms': 'Dark green to black spots on leaves, premature leaf drop',
        'treatment': 'Apply fungicides, improve air circulation, remove infected debris'
    },
    'Black_rot': {
        'description': 'Fungal disease causing black rot on fruit and leaf spots.',
        'symptoms': 'Circular brown spots on leaves, black rotted areas on fruit',
        'treatment': 'Prune infected areas, apply copper-based fungicides'
    },
    'Cedar_apple_rust': {
        'description': 'Fungal disease requiring both apple and cedar trees to complete lifecycle.',
        'symptoms': 'Yellow-orange spots on leaves, deformed fruit',
        'treatment': 'Remove nearby cedar trees, apply protective fungicides'
    },
    'Early_blight': {
        'description': 'Common fungal disease affecting tomatoes and potatoes.',
        'symptoms': 'Dark spots with concentric rings on leaves',
        'treatment': 'Improve air circulation, apply fungicides, crop rotation'
    },
    'Late_blight': {
        'description': 'Serious fungal disease that can destroy entire crops quickly.',
        'symptoms': 'Water-soaked spots on leaves, white fuzzy growth on undersides',
        'treatment': 'Remove infected plants immediately, apply preventive fungicides'
    },
    'Powdery_mildew': {
        'description': 'Fungal disease creating white powdery coating on leaves.',
        'symptoms': 'White powdery patches on leaves and stems',
        'treatment': 'Improve air circulation, apply sulfur-based fungicides'
    },
    'Bacterial_spot': {
        'description': 'Bacterial disease causing spots on leaves and fruit.',
        'symptoms': 'Small dark spots with yellow halos on leaves',
        'treatment': 'Use copper-based bactericides, avoid overhead watering'
    }
}

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        # Update this path to your model location
        model_path = "./model/plant_disease_model.keras"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure the model file 'plant_disease_model.keras' is in the same directory as this script.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (128x128)
        image = image.resize((128, 128))
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.array([img_array])  # Create batch dimension
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def parse_prediction(predicted_class):
    """Parse the predicted class name into components"""
    if "___" in predicted_class:
        plant_type, disease = predicted_class.split("___", 1)
        plant_type = plant_type.replace("_", " ").replace(",", "").title()
        disease = disease.replace("_", " ").title()
        
        if disease.lower() == "healthy":
            status = "Healthy"
            disease = "No disease detected"
        else:
            status = "Diseased"
    else:
        plant_type = predicted_class.replace("_", " ").title()
        disease = "Unknown"
        status = "Unknown"
    
    return plant_type, disease, status

def get_disease_info(disease_name):
    """Get detailed information about a disease"""
    # Clean the disease name for lookup
    clean_name = disease_name.replace(" ", "_").replace("-", "_")
    for key in DISEASE_INFO.keys():
        if key.lower() in clean_name.lower():
            return DISEASE_INFO[key]
    return None

def create_confidence_chart(predictions, class_names, top_n=5):
    """Create a confidence chart for top predictions"""
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_classes = [class_names[i].replace("___", " - ").replace("_", " ") for i in top_indices]
    top_confidences = [predictions[0][i] * 100 for i in top_indices]
    
    # Create color scale - green for highest, fading to orange
    colors = ['#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9']
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_confidences,
            y=top_classes,
            orientation='h',
            marker_color=colors[:len(top_classes)],
            text=[f'{conf:.1f}%' for conf in top_confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Top Predictions Confidence',
        xaxis_title='Confidence (%)',
        yaxis_title='Disease Class',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Plant Disease Detection System</h1>
        <p>Upload an image of a plant leaf to detect diseases using deep learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🔍 Disease Detection", "📊 Model Information", "🌿 Disease Database", "📈 Analytics"]
        )
        
        st.markdown("---")
        st.markdown("## 🏥 Quick Tips")
        st.info("""
        **For best results:**
        - Use clear, well-lit images
        - Focus on the affected leaf area
        - Avoid blurry or dark images
        - Ensure the leaf fills most of the frame
        """)
        
        st.markdown("## 📞 Support")
        st.markdown("""
        - 🔬 Model: CNN with TensorFlow
        - 📸 Input: 128x128 RGB images
        - 🎯 Classes: 38 disease types
        - 📊 Accuracy: High precision detection
        """)
    
    if page == "🔍 Disease Detection":
        detection_page()
    elif page == "📊 Model Information":
        model_info_page()
    elif page == "🌿 Disease Database":
        disease_database_page()
    elif page == "📈 Analytics":
        analytics_page()

def detection_page():
    """Main disease detection page"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.markdown("**Image Information:**")
            st.write(f"- **Filename:** {uploaded_file.name}")
            st.write(f"- **Size:** {image.size}")
            st.write(f"- **Format:** {image.format}")
            st.write(f"- **Mode:** {image.mode}")
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔬 Analysis Results")
            
            # Load model
            model = load_model()
            if model is None:
                st.error("Model could not be loaded. Please check the model file.")
                return
            
            # Predict button
            if st.button("🔍 Analyze Plant Disease", type="primary"):
                with st.spinner("Analyzing image... Please wait."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    if processed_image is None:
                        st.error("Error preprocessing image")
                        return
                    
                    # Make prediction
                    predictions = model.predict(processed_image)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[predicted_class_index]
                    confidence = predictions[0][predicted_class_index]
                    
                    # Parse results
                    plant_type, disease, status = parse_prediction(predicted_class)
                    
                    # Display results
                    result_class = "healthy-result" if status == "Healthy" else "diseased-result"
                    
                    st.markdown(f"""
                    <div class="result-container {result_class}">
                        <h3>🎯 Detection Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Plant Type", plant_type)
                    with col_b:
                        if status == "Healthy":
                            st.metric("Status", "✅ Healthy")
                        else:
                            st.metric("Status", "⚠️ Diseased")
                    with col_c:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Disease information
                    st.markdown("#### 🔍 Detailed Analysis")
                    if status != "Healthy":
                        st.error(f"**Disease Detected:** {disease}")
                        
                        # Get disease information
                        disease_info = get_disease_info(disease)
                        if disease_info:
                            st.markdown("**Disease Information:**")
                            st.info(disease_info['description'])
                            
                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.markdown("**Symptoms:**")
                                st.write(disease_info['symptoms'])
                            with col_y:
                                st.markdown("**Treatment:**")
                                st.write(disease_info['treatment'])
                    else:
                        st.success(f"**Great news!** Your {plant_type} plant appears to be healthy!")
                    
                    # Confidence chart
                    st.markdown("#### 📊 Prediction Confidence")
                    fig = create_confidence_chart(predictions, CLASS_NAMES)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save results to session state for analytics
                    if 'detection_history' not in st.session_state:
                        st.session_state.detection_history = []
                    
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'plant_type': plant_type,
                        'disease': disease,
                        'status': status,
                        'confidence': confidence,
                        'filename': uploaded_file.name
                    })

def model_info_page():
    """Model information page"""
    st.markdown("# 📊 Model Information")
    
    # Load model for info
    model = load_model()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 🧠 Model Architecture")
        if model is not None:
            # Model summary
            st.markdown("### Model Summary")
            try:
                # Get model summary as string
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    model.summary()
                summary_string = f.getvalue()
                
                st.text(summary_string)
                
                # Model metrics
                st.markdown("### 📈 Model Metrics")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Parameters", f"{model.count_params():,}")
                with col_b:
                    st.metric("Input Shape", "128×128×3")
                with col_c:
                    st.metric("Output Classes", len(CLASS_NAMES))
                
            except Exception as e:
                st.error(f"Error displaying model info: {str(e)}")
        else:
            st.error("Model not loaded")
    
    with col2:
        st.markdown("## 🎯 Dataset Information")
        
        # Dataset stats
        plant_counts = {}
        disease_counts = {'Healthy': 0, 'Diseased': 0}
        
        for class_name in CLASS_NAMES:
            if "___" in class_name:
                plant, disease = class_name.split("___", 1)
                plant = plant.replace("_", " ").replace(",", "").title()
                
                if plant not in plant_counts:
                    plant_counts[plant] = 0
                plant_counts[plant] += 1
                
                if disease.lower() == "healthy":
                    disease_counts['Healthy'] += 1
                else:
                    disease_counts['Diseased'] += 1
        
        # Plant distribution chart
        fig_plants = px.bar(
            x=list(plant_counts.keys()),
            y=list(plant_counts.values()),
            title="Plant Types in Dataset",
            labels={'x': 'Plant Type', 'y': 'Number of Classes'},
            color=list(plant_counts.values()),
            color_continuous_scale='Greens'
        )
        fig_plants.update_layout(height=300)
        st.plotly_chart(fig_plants, use_container_width=True)
        
        # Health status distribution
        fig_health = px.pie(
            values=list(disease_counts.values()),
            names=list(disease_counts.keys()),
            title="Health Status Distribution",
            color_discrete_map={'Healthy': '#4CAF50', 'Diseased': '#f44336'}
        )
        fig_health.update_layout(height=300)
        st.plotly_chart(fig_health, use_container_width=True)

def disease_database_page():
    """Disease information database page"""
    st.markdown("# 🌿 Disease Database")
    
    st.markdown("## 🔍 Search Diseases")
    search_term = st.text_input("Search for a disease or plant type...")
    
    # Filter classes based on search
    if search_term:
        filtered_classes = [cls for cls in CLASS_NAMES if search_term.lower() in cls.lower()]
    else:
        filtered_classes = CLASS_NAMES
    
    # Display classes in a nice format
    st.markdown(f"## 📋 Disease Classes ({len(filtered_classes)} found)")
    
    # Group by plant type
    plant_groups = {}
    for class_name in filtered_classes:
        if "___" in class_name:
            plant, disease = class_name.split("___", 1)
            plant = plant.replace("_", " ").replace(",", "").title()
            
            if plant not in plant_groups:
                plant_groups[plant] = []
            plant_groups[plant].append(disease.replace("_", " ").title())
    
    # Display in expandable sections
    for plant, diseases in plant_groups.items():
        with st.expander(f"🌱 {plant} ({len(diseases)} conditions)"):
            for disease in diseases:
                if disease.lower() == "healthy":
                    st.success(f"✅ {disease}")
                else:
                    st.warning(f"⚠️ {disease}")
                    
                    # Show disease info if available
                    disease_info = get_disease_info(disease)
                    if disease_info:
                        st.markdown(f"**Description:** {disease_info['description']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Symptoms:** {disease_info['symptoms']}")
                        with col2:
                            st.markdown(f"**Treatment:** {disease_info['treatment']}")
                    st.markdown("---")

def analytics_page():
    """Analytics and history page"""
    st.markdown("# 📈 Analytics Dashboard")
    
    if 'detection_history' not in st.session_state or not st.session_state.detection_history:
        st.info("No detection history available. Upload and analyze some images first!")
        return
    
    history = st.session_state.detection_history
    df = pd.DataFrame(history)
    
    # Summary metrics
    st.markdown("## 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(history))
    with col2:
        healthy_count = len([h for h in history if h['status'] == 'Healthy'])
        st.metric("Healthy Plants", healthy_count)
    with col3:
        diseased_count = len([h for h in history if h['status'] == 'Diseased'])
        st.metric("Diseased Plants", diseased_count)
    with col4:
        avg_confidence = np.mean([h['confidence'] for h in history])
        st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Plant type distribution
        plant_counts = df['plant_type'].value_counts()
        fig_plants = px.pie(
            values=plant_counts.values,
            names=plant_counts.index,
            title="Plant Types Analyzed"
        )
        st.plotly_chart(fig_plants, use_container_width=True)
    
    with col2:
        # Status distribution
        status_counts = df['status'].value_counts()
        fig_status = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Health Status Distribution",
            color=status_counts.index,
            color_discrete_map={'Healthy': '#4CAF50', 'Diseased': '#f44336'}
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    # Confidence distribution
    st.markdown("### 🎯 Confidence Score Distribution")
    fig_conf = px.histogram(
        df,
        x='confidence',
        nbins=20,
        title="Prediction Confidence Distribution",
        labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
    )
    fig_conf.update_layout(xaxis_tickformat='.1%')
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Detection history table
    st.markdown("### 📋 Detection History")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    
    st.dataframe(
        display_df[['timestamp', 'filename', 'plant_type', 'disease', 'status', 'confidence']],
        use_container_width=True
    )
    
    # Download history as CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name=f"plant_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.detection_history = []
        st.success("History cleared!")
        st.experimental_rerun()

if __name__ == "__main__":
    main()