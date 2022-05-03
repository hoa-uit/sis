import numpy as np
from PIL import Image,ExifTags
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
img_names = []
img_urls = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
    img_names.append(feature_path.stem)
    a = feature_path.stem.rpartition('.')[0].rpartition('.')
    img_urls.append("localhost:3000/" + a[0] + "/" + a[2])
    # print("feature_path.stem: ",feature_path.stem)
    # print("a: ",a)
    # print("a[0]: ",a[0])
    # print("a[2]: ",a[2])

features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

       
                
        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:15]  # Top 15 results  (array ids of image in img folder)
        scores = [(dists[id], img_paths[id], img_names[id], img_urls[id]) for id in ids]   # array save accurate and url/path of image
        
        # exif_data = img._getexif()

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores,
                               img_urls=img_urls,
                               )
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
