import os
from flask import Flask, render_template, redirect, request, session, url_for
import nibabel as nib
from nilearn.image import reorder_img
import tigersyn
import numpy as np
import matplotlib
import cv2
from tigersyn.brainage.utils import get_volumes
from tools import *

matplotlib.use('agg')
app = Flask(__name__)

if not os.path.isdir('uploads'):
    os.mkdir('uploads')
if not os.path.isdir('static'):
    os.mkdir('static')
if not os.path.isdir('download'):
    os.mkdir('download')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'EE304'


@app.route('/')
def home():
    if 'patient_id' in session:
        return redirect(url_for('upload', patient_id=session['patient_id']))
    else:
        return render_template('cover.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        session['patient_id'] = patient_id
        return redirect(url_for('upload', patient_id=patient_id))
    else:
        if 'patient_id' in session:
            return redirect(url_for('upload', patient_id=session['patient_id']))
        else:
            return render_template('login.html')


@app.route('/patient=<patient_id>', methods=['GET', 'POST'])
def upload(patient_id):
    if 'patient_id' in session:
        if request.method == 'POST':
            file = request.files['file']
            if '.nii' in file.filename:
                session['img_fname'] = os.path.basename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
                return redirect(url_for('show', patient_id=session['patient_id']))

        return render_template('upload.html')

    else:
        return redirect(url_for('home'))


@app.route('/logout')
def logout():
    session.pop('patient_id', None)
    return redirect(url_for('home'))


@app.route("/patient=<patient_id>/show", methods=['GET', 'POST'])
def show(patient_id):
    if 'img_fname' in session:
        #img = reorder_img(
        #    nib.load(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz')),
        #    resample='nearest').get_fdata()

        #img_save3GrayScale(img, 'static', 'sagittal.jpg', 'axial.jpg', 'coronal.jpg')

        if request.method == 'POST':
            session['seg_model'] = request.form.get('seg_model')
            return redirect(url_for('segmentation', patient_id=session['patient_id']))
        else:
            raw_img_fname = session['img_fname']
            html1 = f'''
{{% extends "upload.html" %}}

{{% block show_image %}}


  <h1 class="text-center">{raw_img_fname}</h1>
  <p></p>
  <h1 class="text-center">MRI Image</h1>
'''
            html2 = view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'), False)
            html3 = '''
  <hr>
  <form method="POST" class="img-thumbnail">
    <h3>選擇執行的模型：</h3>
    <div class="form-check">
      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault1" value="SynthSeg" checked>
      <label class="form-check-label" for="flexRadioDefault1">
        SynthSeg
      </label>
    </div>
    <div class="form-check">
      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault2" value="Hippocampus">
      <label class="form-check-label" for="flexRadioDefault2">
        Hippocampus
      </label>
    </div>
    <div class="col-12">
      <button type="submit" class="btn btn-primary">確定分割</button>
    </div>
  </form>
  {% block show_mask %}{% endblock %}
{% endblock %}            
'''
            return render_template_string(html1 + html2 + html3)
        # return render_template('display.html', raw_img_fname=session['img_fname'])
    else:
        return redirect(url_for('home'))


@app.route("/patient=<patient_id>/segmentation", methods=['GET', 'POST'])
def segmentation(patient_id):
    if session['seg_model'] == 'SynthSeg':
        print("SynthSeg")
        tigersyn.run('s', os.path.join(app.config['UPLOAD_FOLDER'], '*.nii.gz'), r'static')

        brain_age = tigersyn.predict_age(os.path.join('static', 'image_syn.nii.gz'))  #brain_age
        brain_age = int(round(brain_age, 0))
        brain_size = get_volumes(os.path.join('static', 'image_syn.nii.gz'),
                                 labels)  #each size of label
        brain_size = (np.rint(brain_size)).astype(int)
        brain_sameAgeRange_size = get_All_label_brain_sameAgeRange_size(
        )  #load label average size data
        brain_age_range = get_brain_age_range(brain_age)  #  decision which age class   0~5 x座標index

        raw_img_fname = session['img_fname']
        html1 = f'''
        {{% extends "upload.html" %}}
    
        {{% block show_image %}}
<div class="wrap">
    <div class="left">
          <h1 class="text-center">{raw_img_fname}</h1>
          <p></p>
          <h1 class="text-center">MRI Image</h1>
        '''
        html2 = view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'), False)
        html3 = '''
          <hr>
          <form method="POST" class="img-thumbnail">
            <h3>選擇執行的模型：</h3>
            <div class="form-check">
              <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault1" value="SynthSeg" checked>
              <label class="form-check-label" for="flexRadioDefault1">
                SynthSeg
              </label>
            </div>
            <div class="form-check">
              <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault2" value="Hippocampus">
              <label class="form-check-label" for="flexRadioDefault2">
                Hippocampus
              </label>
            </div>
            <div class="col-12">
              <button type="submit" class="btn btn-primary">確定分割</button>
            </div>
          </form>
        '''
        html4 = view1image(os.path.join('static', 'image_syn.nii.gz'), True)
        html5 = f'''


      
    </div>
    
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 引入Chart.js库 -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* CSS样式来设置图表居中 */
        #myChart {{
            display: block;
            margin: auto;
        }}
        .custom-legend {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-right: 20px;
        }}
        .orange-circle, .blue-circle, .red-circle {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .orange-circle {{
            background-color: red;
        }}
        .blue-circle {{
            background-color: orange;
        }}
        .red-circle {{
            background-color: red;
        }}
        .legend-text {{
            font-size: 24px;
        }}

    </style>


<div class="right">
    <!-- 長條圖顯示的位置 -->
    <canvas id="myChart" width="1200" height="800"></canvas>
    <br>
    <div class="custom-legend">
        
        <div class="legend-item">
            <div class="blue-circle"></div>
            <div class="legend-text" id="blueText">同年齡層腦組織體積為{brain_sameAgeRange_size[0][brain_age_range]}mm^3</div>
        </div>
        <div class="legend-item">
            <div class="orange-circle"></div>
            <div class="legend-text">您的腦齡為{brain_age}歲</div>
        </div>
        <div class="legend-item">
            <div class="red-circle"></div>
            <div class="legend-text" id="redText">您的腦組織體積為{brain_size[0]}mm^3</div>
        </div>
    </div>
    <div>
        <label for="dataSelect">選擇組織：</label>
        <select id="dataSelect">
            <option value="data1">Left Cerebral WM_2</option>
            <option value="data2">Left Cerebral Cortex_3</option>
            <option value="data3">Left Lateral Ventricle_4</option>
            <option value="data4">Left Inf Lat Vent_5</option>
            <option value="data5">Left Cerebellum WM_7</option>
            <option value="data6">Left Cerebellum Cortex_8</option>
            <option value="data7">Left Thalamus_10</option>
            <option value="data8">Left Caudate_11</option>
            <option value="data9">Left Putamen_12</option>
            <option value="data10">Left Pallidum_13</option>
            <option value="data11">3rd Ventricle_14</option>
            <option value="data12">4th Ventricle_15</option>
            <option value="data13">Brain Stem_16</option>
            <option value="data14">Left Hippocampus_17</option>
            <option value="data15">Left Amygdala_18</option>
            <option value="data16">CSF_26</option>
            <option value="data17">Left Accumbens area_28</option>
            <option value="data18">Right Cerebral WM_41</option>
            <option value="data19">Right Cerebral Cortex_42</option>
            <option value="data20">Right Lateral Ventricle_43</option>
            <option value="data21">Right Inf Lat Vent_44</option>
            <option value="data22">Right Cerebellum WM_46</option>
            <option value="data23">Right Cerebellum Cortex_47</option>
            <option value="data24">Right Thalamus_49</option>
            <option value="data25">Right Caudate_50</option>
            <option value="data26">Right Putamen_51</option>
            <option value="data27">Right Pallidum_52</option>
            <option value="data28">Right Hippocampus_53</option>
            <option value="data29">Right Amygdala_54</option>
            <option value="data30">Right Accumbens area_58</option>
            <option value="data31">Right VentralDC_60</option>
            
        </select>
    </div>
    <script>
        const chartElement = document.getElementById('myChart');
        let currentData = [
            {brain_sameAgeRange_size[0]}, // 数据集1
            {brain_sameAgeRange_size[1]}, // 数据集2
            {brain_sameAgeRange_size[2]}, // 数据集3
            {brain_sameAgeRange_size[3]}, // 数据集4
            {brain_sameAgeRange_size[4]}, // 数据集5
            {brain_sameAgeRange_size[5]}, // 数据集6
            {brain_sameAgeRange_size[6]}, // 数据集7
            {brain_sameAgeRange_size[7]}, // 数据集8
            {brain_sameAgeRange_size[8]}, // 数据集9
            {brain_sameAgeRange_size[9]}, // 数据集10
            {brain_sameAgeRange_size[10]}, // 数据集11
            {brain_sameAgeRange_size[11]}, // 数据集12
            {brain_sameAgeRange_size[12]}, // 数据集13
            {brain_sameAgeRange_size[13]}, // 数据集14
            {brain_sameAgeRange_size[14]}, // 数据集15
            {brain_sameAgeRange_size[15]}, // 数据集16
            {brain_sameAgeRange_size[16]}, // 数据集17
            {brain_sameAgeRange_size[17]}, // 数据集18
            {brain_sameAgeRange_size[18]}, // 数据集19
            {brain_sameAgeRange_size[19]}, // 数据集20
            {brain_sameAgeRange_size[20]}, // 数据集21
            {brain_sameAgeRange_size[21]}, // 数据集22
            {brain_sameAgeRange_size[22]}, // 数据集23
            {brain_sameAgeRange_size[23]}, // 数据集24
            {brain_sameAgeRange_size[24]}, // 数据集25
            {brain_sameAgeRange_size[25]}, // 数据集26
            {brain_sameAgeRange_size[26]}, // 数据集27
            {brain_sameAgeRange_size[27]}, // 数据集28
            {brain_sameAgeRange_size[28]}, // 数据集29
            {brain_sameAgeRange_size[29]}, // 数据集30
            {brain_sameAgeRange_size[30]}, // 数据集31
        ];
        const data = {{
            labels: [
                '20-29', '30-39', '40-49', '50-59', '60-69', '70-79'
            ],
            datasets: [{{
                label: '',
                data: currentData[0], // 默认显示数据集1
                borderColor: 'blue',
                pointStyle: 'circle', // 将所有数据点的样式设置为圆圈
                pointRadius: 5, // 设置数据点的半径大小
                pointBackgroundColor: [ 'rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 1)'],
            }},
            {{
                label: '',
                data: {get_diagram_point(brain_size,brain_age_range,0)}, // 单独的数据点
                borderColor: 'transparent ',
                borderWidth: 0,
                pointRadius: 8,
                pointRadius: [0, 0, 0, 0, 0, 0, 0],
                pointHoverRadius: 10,
                pointBackgroundColor: 'red', // 单独数据点的颜色
                pointBorderColor: 'red' // 单独数据点的边框颜色
                
            }}]
        }};
        const myChart = new Chart(chartElement, {{
            type: 'line',
            data: data,
            options: {{
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Age',
                            font: {{
                                size: 24 
                            }}
                        }}
                    }},
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '大腦組織體積(mm^3)',
                            font: {{
                                size: 24 
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: '大腦組織體積',
                        font: {{
                            size: 24 // 设置图表标题字体大小
                        }}
                    }}
                }}
            }}
        }});
        
        // 监听选择菜单变化
        document.getElementById('dataSelect').addEventListener('change', function(event) {{
            const selectedValue = event.target.value;
            // 根据选择的值更新当前数据集
            if (selectedValue === 'data1') {{
                myChart.data.datasets[0].data = currentData[0];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,0)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[0][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[0])}mm^3';
            }} else if (selectedValue === 'data2') {{
                myChart.data.datasets[0].data = currentData[1];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,1)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[1][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[1])}mm^3';
            }} else if (selectedValue === 'data3') {{
                myChart.data.datasets[0].data = currentData[2];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,2)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[2][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[2])}mm^3';
            }} else if (selectedValue === 'data4') {{
                myChart.data.datasets[0].data = currentData[3];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,3)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[3][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[3])}mm^3';
            }} else if (selectedValue === 'data5') {{
                myChart.data.datasets[0].data = currentData[4];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,4)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[4][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[3])}mm^3';
            }} else if (selectedValue === 'data6') {{
                myChart.data.datasets[0].data = currentData[5];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,5)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[5][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[5])}mm^3';
            }} else if (selectedValue === 'data7') {{
                myChart.data.datasets[0].data = currentData[6];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,6)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[6][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[6])}mm^3';
            }} else if (selectedValue === 'data8') {{
                myChart.data.datasets[0].data = currentData[7];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,7)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[7][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[7])}mm^3';
            }} else if (selectedValue === 'data9') {{
                myChart.data.datasets[0].data = currentData[8];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,8)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[8][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[8])}mm^3';
            }} else if (selectedValue === 'data10') {{
                myChart.data.datasets[0].data = currentData[9];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,9)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[9][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[9])}mm^3';
            }} else if (selectedValue === 'data11') {{
                myChart.data.datasets[0].data = currentData[10];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,10)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[10][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[10])}mm^3';
            }} else if (selectedValue === 'data12') {{
                myChart.data.datasets[0].data = currentData[11];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,11)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[11][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[11])}mm^3';
            }} else if (selectedValue === 'data13') {{
                myChart.data.datasets[0].data = currentData[12];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,12)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[12][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[12])}mm^3';
            }} else if (selectedValue === 'data14') {{
                myChart.data.datasets[0].data = currentData[13];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,13)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[13][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[13])}mm^3';
            }} else if (selectedValue === 'data15') {{
                myChart.data.datasets[0].data = currentData[14];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,14)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[14][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[14])}mm^3';
            }} else if (selectedValue === 'data16') {{
                myChart.data.datasets[0].data = currentData[15];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,15)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[15][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[15])}mm^3';
            }} else if (selectedValue === 'data17') {{
                myChart.data.datasets[0].data = currentData[16];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,16)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[16][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[16])}mm^3';
            }} else if (selectedValue === 'data18') {{
                myChart.data.datasets[0].data = currentData[17];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,17)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[17][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[17])}mm^3';
            }} else if (selectedValue === 'data19') {{
                myChart.data.datasets[0].data = currentData[18];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,18)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[18][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[18])}mm^3';
            }} else if (selectedValue === 'data20') {{
                myChart.data.datasets[0].data = currentData[19];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,19)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[19][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[19])}mm^3';
            }} else if (selectedValue === 'data21') {{
                myChart.data.datasets[0].data = currentData[20];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,20)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[20][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[20])}mm^3';
            }} else if (selectedValue === 'data22') {{
                myChart.data.datasets[0].data = currentData[21];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,21)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[21][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[21])}mm^3';
            }} else if (selectedValue === 'data23') {{
                myChart.data.datasets[0].data = currentData[22];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,22)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[22][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[22])}mm^3';
            }} else if (selectedValue === 'data24') {{
                myChart.data.datasets[0].data = currentData[23];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,23)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[23][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[23])}mm^3';
            }} else if (selectedValue === 'data25') {{
                myChart.data.datasets[0].data = currentData[24];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,24)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[24][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[24])}mm^3';
            }} else if (selectedValue === 'data26') {{
                myChart.data.datasets[0].data = currentData[25];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,25)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[25][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[25])}mm^3';
            }} else if (selectedValue === 'data27') {{
                myChart.data.datasets[0].data = currentData[26];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,26)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[26][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[26])}mm^3';
            }} else if (selectedValue === 'data28') {{
                myChart.data.datasets[0].data = currentData[27];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,27)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[27][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[27])}mm^3';
            }} else if (selectedValue === 'data29') {{
                myChart.data.datasets[0].data = currentData[28];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,28)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[28][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[28])}mm^3';
            }} else if (selectedValue === 'data30') {{
                myChart.data.datasets[0].data = currentData[29];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,29)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[29][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[29])}mm^3';
            }} else if (selectedValue === 'data31') {{
                myChart.data.datasets[0].data = currentData[30];
                myChart.data.datasets[1].data = {get_diagram_point(brain_size,brain_age_range,30)};
                document.getElementById('blueText').textContent = '同年齡層腦組織體積為{str(brain_sameAgeRange_size[30][brain_age_range])}mm^3';
                document.getElementById('redText').textContent = '你的腦組織體積為{str(brain_size[30])}mm^3';
            }}
            // 更新图表显示
            myChart.update();
        }});
        function createSeparateDataPoint(data, index,bar) {{
            const separateData = Array(6).fill(null);
            separateData[index] = data[bar];
            return separateData;
        }}
        // 指定特定 x 轴索引的数据点变为红色
        const targetIndex = {brain_age_range}; // 指定 x 轴索引（从0开始）
        myChart.data.datasets[0].pointBackgroundColor[targetIndex] = 'orange';
        myChart.data.datasets[1].pointRadius[targetIndex] = 10;
        myChart.update();
    </script>


    </div>
</div>
<style>
  .wrap{{
    width: 100%;
    height: 900px;
    display: flex;
    justify-content: space-between;
}}
.left{{
    background: white;
    width: 40%;
    height: 800px;
}}
.right{{
    background: white;
    width: 60%;
    height: 900px;
}}
</style>

        '''
        html6 = '''
          {% block show_mask %}{% endblock %}
        {% endblock %}            
        '''
        return render_template_string(html1 + html2 + html3 + html4 + html5 + html6)
    #return render_template('segmentation.html', raw_img_fname=session['img_fname'])
    if session['seg_model'] == 'Hippocampus':
        print("Hippocampus")

        tigersyn.run('h', os.path.join(app.config['UPLOAD_FOLDER'], '*.nii.gz'), r'static')
        brain_size = get_volumes(os.path.join('static', 'image_hippocampus.nii.gz'),
                                 [1])  # each size of label
        brain_size = (np.rint(brain_size)).astype(int)

        raw_img_fname = session['img_fname']
        html1 = f'''
                {{% extends "upload.html" %}}

                {{% block show_image %}}


                  <h1 class="text-center">{raw_img_fname}</h1>
                  <p></p>
                  <h1 class="text-center">MRI Image</h1>
                '''
        html2 = view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'), False)
        html3 = '''
                  <hr>
                  <form method="POST" class="img-thumbnail">
                    <h3>選擇執行的模型：</h3>
                    <div class="form-check">
                      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault1" value="SynthSeg" checked>
                      <label class="form-check-label" for="flexRadioDefault1">
                        SynthSeg
                      </label>
                    </div>
                    <div class="form-check">
                      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault2" value="Hippocampus">
                      <label class="form-check-label" for="flexRadioDefault2">
                        Hippocampus
                      </label>
                    </div>
                    <div class="col-12">
                      <button type="submit" class="btn btn-primary">確定分割</button>
                    </div>
                  </form>
                '''
        html4 = view1image(os.path.join('static', 'image_hippocampus.nii.gz'),
                           False)  #view1image(os.path.join('static', "hippo.onnx_image.nii.gz"))
        html5 = f'''
                <style>
                    .ball {{
                      width: 20px;
                      height: 20px;
                      background-color: red;
                      border-radius: 50%; /* 使其成為圓形 */
                      position: relative;
                    }}
                    .text {{
                      position: absolute;
                      top: 50%;
                      transform: translateY(-50%);
                      left: 40px; /* 調整文字與圓球的距離 */
                      font-size: 24px;
                      color: black;
                      white-space: nowrap;
                    }}
                </style>
                <div style="display: flex; justify-content: center;">
                <div class="ball">
                    <p class="text">您的海馬迴體積為{str(brain_size[0])}mm^3</p>
                </div>
                </div>
                  {{% block show_mask %}}{{% endblock %}}
                {{% endblock %}}
                '''
        return render_template_string(html1 + html2 + html3 + html4 + html5)


if __name__ == '__main__':
    app.run(debug=True)
