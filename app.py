import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import nibabel as nib
import cv2
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
import os
from inference_opt import evaluate_model_accuracy, accuracy_indicator_fig

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "models/category_model1.h5"
IMG_SIZE = 128
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Labels used during training
LABELS = ["glioma", "meningioma" ,"pituitary", "no_tumor"]

# -------------------------------
# DASH APP SETUP
# -------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Brain Tumor Category Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Brain Tumor Category Prediction Dashboard"), width=12)
    ], className="my-3"),
    
    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a .nii MRI File')
                ]),
                style={
                    'width': '100%', 'height': '100px', 'lineHeight': '100px',
                    'borderWidth': '2px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            width=6
        ),
        dbc.Col(
            html.Div(id='output-prediction', style={'fontSize': 20, 'fontWeight': 'bold'}),
            width=6
        )
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='slice-graph'), width=6),
        dbc.Col(dcc.Graph(id='prob-bar'), width=6)
    ])
], fluid=True)

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def preprocess_nii(file_path):
    # Load NIfTI
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()

    # Extract middle slice
    mid_slice = img_data.shape[2] // 2
    img_2d = img_data[:, :, mid_slice]

    # Resize and normalize
    img_2d = cv2.resize(img_2d, (IMG_SIZE, IMG_SIZE))
    img_2d_norm = img_2d / 255.0
    img_2d_input = np.expand_dims(img_2d_norm, axis=-1)  # add channel
    img_2d_input = np.expand_dims(img_2d_input, axis=0)   # add batch
    return img_2d_norm, img_2d_input

def create_slice_figure(slice_2d):
    fig = px.imshow(slice_2d, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0))
    return fig

def create_prob_bar(pred_probs):
    fig = px.bar(x=LABELS, y=pred_probs[0], labels={'x':'Category', 'y':'Probability'}, text=pred_probs[0])
    fig.update_yaxes(range=[0,1])
    return fig

# -------------------------------
# CALLBACK
# -------------------------------
@app.callback(
    Output('output-prediction', 'children'),
    Output('slice-graph', 'figure'),
    Output('prob-bar', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        # Save uploaded file temporarily
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(save_path, "wb") as f:
            f.write(decoded)

        # Preprocess
        slice_2d, input_img = preprocess_nii(save_path)

        # Predict
        pred_probs = model.predict(input_img)
        class_idx = np.argmax(pred_probs)
        pred_label = LABELS[class_idx]
        confidence = pred_probs[0][class_idx]

        # Create figures
        slice_fig = create_slice_figure(slice_2d)
        prob_fig = create_prob_bar(pred_probs)

        return f"Predicted Category: {pred_label} (Confidence: {confidence:.2f})", slice_fig, prob_fig
    else:
        return "Upload a .nii MRI file to predict.", px.imshow(np.zeros((IMG_SIZE, IMG_SIZE))), px.bar(x=LABELS, y=[0,0,0,0])

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)

acc, total, correct = evaluate_model_accuracy(model, "test_labels.csv", LABELS, img_size=IMG_SIZE, num_slices=5, do_tta=True)
print("Accuracy:", acc, "total:", total, "correct:", correct)
fig = accuracy_indicator_fig(acc)   # show in Dash or fig.show()