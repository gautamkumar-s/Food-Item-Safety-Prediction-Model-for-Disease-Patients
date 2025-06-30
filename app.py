from flask import Flask, render_template, request
from dia import predict_diabetes, predict_bp, predict_obesity

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = ""
    if request.method == 'POST':
        try:
            disease = request.form['disease']

            
            if disease == 'Diabetes':
                required_fields = ['kcal', 'protein', 'fat', 'saturated_fat', 'carbs', 'sugar']
                input_data = [float(request.form[field]) for field in required_fields]
                result = predict_diabetes(input_data)
                
                output = "This is GOOD for diabetic patient." if result == 1 else "This is NOT GOOD for diabetic patient."

            elif disease == 'highbloodpressure':
                required_fields = ['saturated_fat', 'sugar', 'calcium', 'magnesium', 'potassium', 'sodium']
                input_data = [float(request.form[field]) for field in required_fields]
                result = predict_bp(input_data)
                output = "This is GOOD for high blood pressure." if result == 1 else "This is NOT GOOD for high blood pressure."

            elif disease == 'obesity':
                required_fields = ['kcal', 'saturated_fat', 'fat', 'carbs', 'sugar', 'calcium', 'iron', 'magnesium', 'sodium']
                input_data = [float(request.form[field]) for field in required_fields]
                result = predict_obesity(input_data)
                output = "This is GOOD for obese patient." if result == 1 else "This is NOT GOOD for obese patient."

            else:
                output = "Error: Unknown disease selected."

        except ValueError:
            output = "Error: One or more required fields are missing or contain non-numeric values."

        except Exception as e:
            output = f"Unexpected error: {str(e)}"

    return render_template('dia.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
