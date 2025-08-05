from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("laptop_price_model.pkl")

# Mapping from categories to numeric values (from your dataset)
brand_mapping = {
    'HP': 0, 'Dell': 1, 'Lenovo': 2, 'Asus': 3, 'Acer': 4,
    'MSI': 5, 'Apple': 6, 'Avita': 7, 'Infinix': 8, 'Samsung': 9
}
processor_name_mapping = {
    'Intel Core i3': 0, 'Intel Core i5': 1, 'Intel Core i7': 2,
    'AMD Ryzen 5': 3, 'AMD Ryzen 7': 4, 'Other': 5
}
processor_gen_mapping = {
    '4th': 0, '5th': 1, '6th': 2, '7th': 3, '8th': 4,
    '9th': 5, '10th': 6, '11th': 7, '12th': 8
}

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            brand = brand_mapping.get(request.form["brand"], 0)
            processor_name = processor_name_mapping.get(request.form["processor_name"], 0)
            processor_gnrtn = processor_gen_mapping.get(request.form["processor_gnrtn"], 0)

            ram_gb = int(request.form["ram_gb"])
            ssd = int(request.form["ssd"])
            hdd = int(request.form["hdd"])
            graphic_card_gb = int(request.form["graphic_card_gb"])
            touchscreen = int(request.form["Touchscreen"])
            msoffice = int(request.form["msoffice"])

            features = np.array([[brand, processor_name, processor_gnrtn,
                                  ram_gb, ssd, hdd, graphic_card_gb,
                                  touchscreen, msoffice]])

            prediction = round(model.predict(features)[0], 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


