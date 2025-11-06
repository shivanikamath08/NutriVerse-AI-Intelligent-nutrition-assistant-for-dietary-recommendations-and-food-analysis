import os
import csv
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
INGREDIENTS_CSV = 'merged_dataset.csv'
NUTRITION_CSV = 'merged_food_data_with_values.csv'


app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MODEL_PATH = 'yolo_results/best.pt'


try:
    yolo_model = YOLO(MODEL_PATH)
except Exception:
    from ultralytics import YOLO as DefaultYOLO
    yolo_model = DefaultYOLO('yolo11n.pt')


def safe_float(val, default=0):
    try:
        return float(val)
    except:
        return default


classes = {
    0:'Biryani',1:'Burger',2:'Dhokla',3:'Donut',4:'Gravy',
    5:'Idli',6:'Jalebi',7:'Roll',8:'Roti',9:'Kulfi'
}


dish_ingredients = {}
def load_ingredients():
    global dish_ingredients
    dish_ingredients = {}
    with open(INGREDIENTS_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        current_dish = None
        for row in reader:
            fname = row['food_name'].strip()
            if fname:
                current_dish = fname
                dish_ingredients[current_dish] = {'main': [], 'additional': []}
            iname = row['ingredient_name'].strip()
            itype = row.get('type','main').lower()
            qty = safe_float(row.get('std_quantity'))
            if current_dish and iname:
                dish_ingredients[current_dish][itype].append({'name': iname, 'grams': qty, 'unit': row.get('std_unit','')})


nutrition_db = {}
def load_nutrition():
    global nutrition_db
    nutrition_db = {}
    with open(NUTRITION_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc = row['description'].strip().lower()
            nutrition_db[desc] = {
                'calories': safe_float(row.get('calorie_value')),
                'protein': safe_float(row.get('protein_value')),
                'carbs': safe_float(row.get('carbohydrate_value')),
                'fat': safe_float(row.get('fat'))
            }


def get_nutrition(name):
    n = name.lower()
    if n in nutrition_db:
        return nutrition_db[n]
    for k in nutrition_db:
        if k in n or n in k:
            return nutrition_db[k]
    # Return minimal zero values if not found
    return {'calories':0,'protein':0,'carbs':0,'fat':0}


load_ingredients()
load_nutrition()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


def detect_foods(img_path):
    try:
        results = yolo_model(img_path)
        detections = []
        for res in results:
            if not hasattr(res, 'boxes') or res.boxes is None:
                continue
            for box in res.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Lower confidence threshold from 0.5 to 0.3 to allow multiple detections
                if conf < 0.3:
                    continue
                cname = classes.get(cls)
                if not cname:
                    continue
                detections.append({'name': cname, 'confidence': conf})
        return detections
    except Exception as ex:
        print(f"Detection exception: {ex}")
        return []


@app.route('/upload_meal', methods=['POST'])
def upload_meal():
    if 'mealImage' not in request.files:
        return jsonify(error='No image uploaded'), 400
    file = request.files['mealImage']
    if not allowed_file(file.filename):
        return jsonify(error='Unsupported image type'), 400
    fname = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, datetime.now().strftime('%Y%m%d%H%M%S_') + fname)
    file.save(save_path)

    detections = detect_foods(save_path)
    if not detections:
        return jsonify(error='No food detected', image_url=f'/uploads/{fname}'), 400

    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    main_dish = sorted_detections[0]['name']

    unique_dishes = sorted(set(d['name'] for d in detections))
    ing_seen = set()
    main_ings = []
    additional_ings = []
    for dish in unique_dishes:
        for ing in dish_ingredients.get(dish, {}).get('main', []):
            if ing['name'] not in ing_seen:
                ing_seen.add(ing['name'])
                main_ings.append({'name': ing['name'], 'grams': ing['grams']})
        for ing in dish_ingredients.get(dish, {}).get('additional', []):
            if ing['name'] not in ing_seen:
                ing_seen.add(ing['name'])
                additional_ings.append({'name': ing['name'], 'grams': 0})

    return jsonify(dish=main_dish,
                   all_dishes=', '.join(unique_dishes),
                   image_url=f'/uploads/{fname}',
                   defaults=main_ings,
                   additionals=additional_ings,
                   confidence=sorted_detections[0]['confidence'],
                   suggestions=[])


@app.route('/calculate_nutrition', methods=['POST'])
def calculate_nutrition():
    try:
        data = request.get_json()
        ingredients = data.get('ingredients', [])
        total = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
        for ing in ingredients:
            n = ing.get('name','').lower()
            g = float(ing.get('grams',0))
            nut = get_nutrition(n)
            protein = nut['protein']*g/100
            carbs = nut['carbs']*g/100
            fat = nut['fat']*g/100
            calories = nut['calories']*g/100
            # fallback calculation if calorie missing or zero
            if not calories or calories <= 0:
                calories = protein*4 + carbs*4 + fat*9
            total['calories'] += calories
            total['protein'] += protein
            total['carbs'] += carbs
            total['fat'] += fat

        rounded = {k: round(v,1) for k,v in total.items()}

        print(f"DEBUG: Nutritional totals: {rounded}")

        suggestions = []

        # Portion size warnings
        total_grams = sum(ing.get('grams',0) for ing in ingredients)
        if total_grams > 1000:
            suggestions.append("Warning: Extremely large portion size.")

        # Nutritional thresholds with more granular checks
        if rounded['calories'] > 800:
            suggestions.append("High calorie meal. Consider portion control.")
        elif rounded['calories'] < 300:
            suggestions.append("Low calorie meal. Ensure sufficient energy intake.")

        if rounded['protein'] < 15:
            suggestions.append("Low protein content. Add protein rich foods.")
        elif rounded['protein'] > 70:
            suggestions.append("High protein content. Balance with other macronutrients.")

        if rounded['carbs'] > 150:
            suggestions.append("High carbohydrate content. Balance your meal.")
        elif rounded['carbs'] < 20:
            suggestions.append("Low carbohydrate content. Include wholesome carbs.")

        if rounded['fat'] > 50:
            suggestions.append("High fat content. Try low-fat alternatives.")
        elif rounded['fat'] < 10:
            suggestions.append("Low fat content. Include healthy fats for balance.")

        if not suggestions:
            suggestions.append("Your meal looks balanced!")

        return jsonify(totals=rounded, suggestions=suggestions)

    except Exception as e:
        print(f"Nutrition calculation error: {e}")
        return jsonify(error='Failed to calculate nutrition'), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
