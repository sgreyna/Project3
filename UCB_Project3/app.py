

import os
from flask import Flask, request, jsonify, render_template
import pandas as pd



app = Flask(__name__)

@app.route("/")
def main():
    return render_template ('home.html')

@app.route('/data')
def getDataFromCSV():
    
    # Create a reference the CSV file desired
    csv_path = "dailyData/dailyData.csv"

    # Read the CSV into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    return render_template ('data.html', tables=[df.to_html(classes = 'data')], titles = df.columns.values)



# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     data = {"success": False}
#     if request.method == 'POST':
#         print(request)

#         if request.files.get('file'):
#             # read the file
#             file = request.files['file']

#             # read the filename
#             filename = file.filename

#             # create a path to the uploads folder
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#             # Save the file to the uploads folder
#             file.save(filepath)

#             # Load the saved image using Keras and resize it to the mnist
#             # format of 28x28 pixels
#             image_size = (28, 28)
#             im = image.load_img(filepath, target_size=image_size,
#                                 grayscale=True)

#             # Convert the 2D image to an array of pixel values
#             image_array = prepare_image(im)
#             print(image_array)

#             # Get the tensorflow default graph and use it to make predictions
#             global graph
#             with graph.as_default():

#                 # Use the model to make a prediction
#                 predicted_digit = model.predict_classes(image_array)[0]
#                 data["prediction"] = str(predicted_digit)

#                 # indicate that the request was a success
#                 data["success"] = True

#             return jsonify(data)
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <p><input type=file name=file>
#          <input type=submit value=Upload>
#     </form>
#     '''


if __name__ == "__main__":
    app.run(debug=True)
