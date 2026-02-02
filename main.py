from flask import Flask, render_template, request
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import StudentClass, PredictPipeline
from src.logger import logger

application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint',methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = StudentClass(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        df = data.get_data_as_data_frame()
        logger.info(df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(df)
        return render_template('home.html', results=results[0])

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.perform_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.perform_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.perform_model_training(train_arr, test_arr)
    app.run(host='0.0.0.0', port=8081)