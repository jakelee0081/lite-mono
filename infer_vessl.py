import test_simple
import streamlit as st
import os
from PIL import Image

st.set_page_config(layout="wide")

args = test_simple.parse_args()

args.load_weights_folder = "./weights"
args.image_path = "./sample_data/0000002426.png"
print(args)
test_simple.test_simple(args)

col1, col2, col3 = st.columns(3)

with col1:
    input_img = Image.open(args.image_path)
    st.image(input_img)
with col2:
    out_img_name = os.path.basename(args.image_path).replace(".png","_disp.jpeg")
    out_img_path = os.path.join(os.path.dirname(args.image_path), out_img_name)
    out_img = Image.open(out_img_path)
    st.image(out_img)

with col3:
    yaml = """name : lite-mono
        resources:
        cluster: aws-apne2-prod1
        accelerators: V100:1
        image: quay.io/vessl-ai/ngc-pytorch-kernel:22.12-py3-202301160809
        run:
        - workdir: /root
            command: |
            git clone https://github.com/jakelee0081/lite-mono.git
            cd lite-mono
            pip install -r requirements.txt
            streamlit run infer_vessl.py -- --load_weights_folder ./weights --image_path ./sample_data/0000002426.png

        runtime: 24h
        ports:
            - 8501
        """
    st.markdown(
        f'<p style="font-family:Courier; color:Black; font-size: 20px;">YAML</p>',
        unsafe_allow_html=True,
    )
    st.code(yaml, language="yaml", line_numbers=False)        