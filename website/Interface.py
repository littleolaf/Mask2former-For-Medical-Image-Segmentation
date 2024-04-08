import streamlit as st
import subprocess
import os

input_img = ""
with st.sidebar:
    st.title("Image Input")
    uploaded_file = st.file_uploader(label="Please input a img.")
    if uploaded_file:
        file_name = uploaded_file.name
        input_img = os.path.join("/home/tangwuyang/MINImask2former/website/img/input/", file_name)
        with open(input_img, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File Saved Successfully!")
st.title("唐武阳的毕设展示")
with st.expander("模型结果展示"):
    st.write("## 原始图片展示：")
    if uploaded_file:
        st.image(uploaded_file)

    st.write("## 分割结果图片展示：")

    output_path = "/home/tangwuyang/MINImask2former/website/img/output/"
    if uploaded_file:
        command = [
            "python", "/home/tangwuyang/MINImask2former/demo/demo.py",
            "--config-file", "/home/tangwuyang/MINImask2former/configs/kvasir_seg/MaskFormer2_R50_bs16_160k.yaml",
            "--input", f"{input_img}",
            "--output", f"{output_path}",
            "--opts", "MODEL.WEIGHTS", "/home/tangwuyang/MINImask2former/save_model/kvasir_2w/model_0019999.pth"
        ]
        subprocess.run(command)
        output_img = os.path.join(output_path, file_name)
        st.image(output_img)
