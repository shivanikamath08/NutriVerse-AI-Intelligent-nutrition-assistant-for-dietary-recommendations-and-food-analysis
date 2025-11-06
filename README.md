# NutriVerse-AI-Intelligent-nutrition-assistant-for-dietary-recommendations-and-food-analysis
NutriVerse AI is an AI-powered web app that identifies food items from images, calculates nutritional values, and suggests healthier alternatives. It helps users track diet, avoid allergens, and receive personalized recommendations for better health and disease management.

📖 Overview

In today’s fast-paced lifestyle, the consumption of unhealthy food has significantly increased, leading to diet-related health issues such as obesity, diabetes, and cardiovascular diseases. NutriVerse AI addresses this problem by developing an intelligent assistant that leverages computer vision and conversational AI to provide personalized, health-based food suggestions.

🧠 Research Summary

The application utilizes the YOLOv11 architecture integrated with EfficientNetB0 for accurate food detection and classification. The model trains on a curated subset of the Foodies Challenge Dataset, featuring a diverse and balanced selection of Indo-Western cuisines.
-> YOLOv11 performs multi-class object detection using Convolutional Neural Network (CNN) layers.
-> EfficientNetB0 enhances refined classification through optimized feature extraction.


💬 Conversational AI Integration

-> An AI-based chatbot is embedded within the system to:
-> Provide alternative dietary suggestions
-> Offer caloric breakdowns
-> Trigger health alerts based on user conditions

🍽️ Nutrient Computation

Users manually input approximate portion sizes of each ingredient present in their meal. The system calculates macronutrient content (Carbohydrates, Proteins, and Fats per 100g) using the USDA Foundational Food Dataset, estimates total calorie intake, and identifies nutrient imbalances.


🌐 Frontend
-> Technologies: HTML5, CSS3, JavaScript (ES6+)
-> Visualization: Chart.js for interactive pie charts and nutrition insights
-> Design: Google Fonts and Font Awesome for fonts and icons

🖥️ Backend
-> Programming Language: Python 3.x
-> Framework: Flask for constructing RESTful APIs

Libraries & Tools:
-> PyTorch (Torch): For loading and running the YOLOv11 object detection model
-> Pandas / CSV Module: For managing and processing nutritional databases (CSV files)
-> CORS (Cross-Origin Resource Sharing): Enabled for smooth client–server communication

🧠 AI/ML Models
-> YOLOv11 + EfficientNetB0 for multi-class food detection and classification
