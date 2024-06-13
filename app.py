import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import numpy as np
import datetime 
from math import floor
import os
import json
import requests
from streamlit_lottie import st_lottie

date = datetime.datetime.now()
today_file = f'Data/user_data_{date.day}{date.month}{date.year}.csv'

try:
    st.session_state.user_data = pd.read_csv(today_file)
except FileNotFoundError:
    data = {
    "exercise": ["Lunge", "Push Up", "Squat"],
    "correct_count": [0, 0, 0],
    "incorrect_count": [0, 0, 0],
    "sets": [0, 0, 0],
    "date": [None, None, None]
}
    df = pd.DataFrame(data)
    df.to_csv(f'Data/user_data_{date.day}{date.month}{date.year}.csv', index=False)
    st.session_state.user_data = pd.read_csv(f'Data/user_data_{date.day}{date.month}{date.year}.csv')

def today_date():
    date = datetime.datetime.now()
    st.session_state.user_data['date'] = f'{date.day}-{date.month}-{date.year}'
    st.session_state.user_data.to_csv(f'Data/user_data_{date.day}{date.month}{date.year}.csv', index=False)

today_date()

def clean_df(df):# SULTAAAAAAAAAN 
    cols = df.select_dtypes(include='number').columns.tolist()
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y').dt.date
    df = df[cols].astype(int)

def update_df(exercise, col, value, assign=None):
    if assign:
        st.session_state.user_data.loc[st.session_state.user_data['exercise'] == exercise, col] = value
    else:
        st.session_state.user_data.loc[st.session_state.user_data['exercise'] == exercise, col] += value
    st.session_state.user_data.to_csv(f'Data/user_data_{date.day}{date.month}{date.year}.csv', index=False)

# Load reference images for exercises
reference_images = {
    'Squat': 'images/correct_squat_image.jpg',
    'Squat Front': 'images/squat_front.jpg',
    'Push Up': 'images/correct_pushup_image.png',
    'Lunge': 'images/correct_lunge_image.png'
}

# Function to load and resize images
def load_and_resize(image_path, size=(320, 240)):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        # Resize the image with interpolation for better quality
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return image

# Load all reference images
loaded_images = {name: load_and_resize(path) for name, path in reference_images.items()}

# Setup MediaPipe instance
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize counters and flags
correct_attempts = 0
exercise_started = False
correct_exercise = False



def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_exercise(exercise):
    global correct_attempts, exercise_started, correct_exercise

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and get pose landmarks
        results = pose.process(image_rgb)

        # Draw the pose landmarks on the frame
        correctness = ''
        color = None
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get the landmark positions
            landmarks = results.pose_landmarks.landmark

            # Check correctness of exercise pose based on angles
            if exercise == 'Squat':
                keypoints = [mp_pose.PoseLandmark.LEFT_HIP,
                             mp_pose.PoseLandmark.LEFT_KNEE,
                             mp_pose.PoseLandmark.LEFT_ANKLE,
                             mp_pose.PoseLandmark.RIGHT_HIP,
                             mp_pose.PoseLandmark.RIGHT_KNEE,
                             mp_pose.PoseLandmark.RIGHT_ANKLE]
                threshold = (70, 110)
                

            elif exercise == 'Push Up':
                keypoints = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                             mp_pose.PoseLandmark.LEFT_ELBOW,
                             mp_pose.PoseLandmark.LEFT_WRIST,
                             mp_pose.PoseLandmark.RIGHT_SHOULDER,
                             mp_pose.PoseLandmark.RIGHT_ELBOW,
                             mp_pose.PoseLandmark.RIGHT_WRIST]
                threshold = (160, 180)
                
            elif exercise == 'Lunge':
                keypoints = [mp_pose.PoseLandmark.LEFT_HIP,
                             mp_pose.PoseLandmark.LEFT_KNEE,
                             mp_pose.PoseLandmark.LEFT_ANKLE,
                             mp_pose.PoseLandmark.RIGHT_HIP,
                             mp_pose.PoseLandmark.RIGHT_KNEE,
                             mp_pose.PoseLandmark.RIGHT_ANKLE]
                threshold = (70, 110)
                

            # Calculate angles and determine correctness
            angles = []
            for i in range(0, len(keypoints) - 2, 3):
                a = (landmarks[keypoints[i]].x, landmarks[keypoints[i]].y)
                b = (landmarks[keypoints[i + 1]].x, landmarks[keypoints[i + 1]].y)
                c = (landmarks[keypoints[i + 2]].x, landmarks[keypoints[i + 2]].y)
                angle = calculate_angle(a, b, c)
                angles.append(angle)
            correctness = "Incorrect"
            color = (0, 0, 255)  # Red color for incorrect
            if len(angles) == 2 and all(threshold[0] <= angle <= threshold[1] for angle in angles):  # Example thresholds for each exercise
                correctness = "Correct"
                color = (0, 255, 0)  # Green color for correct

            if correctness == "Correct":
                if not correct_exercise:
                    if not exercise_started:
                        exercise_started = True  # Start exercise detection
                         # SULTAAAAAAAAAN
                        # st.write(start_time) 
                    correct_attempts += 1  # Increment correct attempts
                    update_df(exercise, 'correct_count', 1) 
                    time.sleep(1)# SULTAAAAAAAAAN 
                    correct_exercise = True
            else:
                correct_exercise = False
                
        
        # Display correctness and attempts on the frame
        cv2.putText(frame, f"Status: {correctness}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Correct Attempts: {correct_attempts}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame in Streamlit
        stframe.image(frame, channels='BGR')
        n_counts = st.session_state.user_data[st.session_state.user_data['exercise'] == exercise]['correct_count'].values[0]
        # st.write(f'{n_counts}/ 4')
        update_df(exercise, 'sets', int(n_counts/4), 'y')
    
st.title("🏋️ Exercise Detection")
st.write("Ensure your webcam is enabled and select an exercise from the sidebar to begin detecting.")

# Sidebar with exercise options
exercise_option = st.sidebar.radio('Select an exercise:', ['Home', 'Squat 🏋️', 'Push Up 💪', 'Lunge 🏃'])

# Load and display Lottie animations for each exercise
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_Squat = load_lottiefile("json/squat.json")
lottie_lunge = load_lottiefile("json/lunge.json")
lottie_push_up = load_lottiefile("json/push_up.json")
lottie_ax = load_lottieurl('https://lottie.host/dd18f32c-9a9a-4c93-985c-dfacc03e80c0/fnzoa3AmYO.json')

# Display Lottie animations in sidebar
with st.sidebar:
    if exercise_option == "Squat 🏋️":
        st_lottie(lottie_lunge, key='lottie_lunge')
        
    elif exercise_option == "Push Up 💪":
        st_lottie(lottie_push_up, key='lottie_push_up')
    elif exercise_option == "Lunge 🏃":
        st_lottie(lottie_Squat, key='lottie_Squat')
if exercise_option == "Home":
    st.write("Welcome to the Exercise Detection App!")
    st.write("This app uses your webcam to detect and analyze your exercises. Select an exercise from the sidebar to get started.")
    st_lottie(lottie_ax, key="user")
else:
    exercise_option = exercise_option.split(" ")[0]    

# Display reference images for the selected exercise
if exercise_option in loaded_images:
    reference_image = loaded_images[exercise_option]

if exercise_option != 'Home':
    # Button to start the exercise detection
    if st.button('Start'):
        btn = st.button('stop exercise')
        if btn:
            # cap.release()
            cv2.destroyAllWindows()
        detect_exercise(exercise_option)

        
        


    # dashboard (EDA)
    files = [file for file in os.listdir('Data/') if file.endswith('.csv') and file != today_file and file != 'aggregated_user_data.csv']
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join('Data/', file))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('Data/aggregated_user_data.csv')

    # st.dataframe(st.session_state.user_data) #dbug
    # st.write(st.session_state.user_data.dtypes) #dbug
    clean_df(st.session_state.user_data)
    clean_df(df)



    def create_dashboard(selected_exercise, date_range):
        filtered_df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
        
        if selected_exercise != 'All':
            filtered_df = filtered_df[filtered_df['exercise'] == selected_exercise]

        # Aggregate data
        total_exercises = filtered_df['exercise'].nunique()
        correct_incorrect_counts = filtered_df.groupby('exercise')[['correct_count', 'incorrect_count']].sum().reset_index()
        sets_distribution = filtered_df.groupby('exercise')['sets'].sum().reset_index()
        daily_performance = filtered_df.groupby('date')[['correct_count', 'incorrect_count']].sum().reset_index()
        
        # Additional metrics
        average_correct = filtered_df['correct_count'].mean()
        average_incorrect = filtered_df['incorrect_count'].mean()
        correct_incorrect_ratio = filtered_df.groupby('exercise').apply(
            lambda x: (x['correct_count'].sum() / (x['correct_count'].sum() + x['incorrect_count'].sum())) * 100).reset_index(name='correct_ratio')

            # Create visualizations
        figures = []
        max_value = max(average_correct, average_incorrect)

        # Correct and incorrect repetitions for each exercise
        fig_correct_incorrect = px.bar(
            correct_incorrect_counts, 
            x='exercise', 
            y=['correct_count', 'incorrect_count'],
            title='Correct and Incorrect Repetitions for Each Exercise',
            labels={'value': 'Count', 'exercise': 'Exercise'},
            barmode='group',
            template='plotly_white',
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        figures.append(fig_correct_incorrect)

        # Distribution of sets performed for each exercise
        if selected_exercise == 'All':
            fig_sets_distribution = px.pie(
                sets_distribution, 
                values='sets', 
                names='exercise',
                title='Distribution of Sets Performed for Each Exercise',
                template='plotly_white',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            figures.append(fig_sets_distribution)

        # Daily performance trends
        fig_daily_performance = px.line(
            daily_performance, 
            x='date', 
            y=['correct_count', 'incorrect_count'],
            title='Daily Performance Trends',
            labels={'value': 'Count', 'date': 'Date'},
            template='plotly_white',
            color_discrete_sequence=['#00CC96', '#AB63FA']
        )
        figures.append(fig_daily_performance)

        # Average correct and incorrect counts per session
        fig_average_correct = go.Figure(go.Indicator(
            mode="gauge+number",
            value=average_correct,
            title={"text": "Average Correct Repetitions Per Session", "font": {"size": 18}},
            gauge={"axis": {"range": [None, max_value], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "green"}},
            number={"font": {"size": 36, "color": "green"}}
        ))
        figures.append(fig_average_correct)

        fig_average_incorrect = go.Figure(go.Indicator(
            mode="gauge+number",
            value=average_incorrect,
            title={"text": "Average Incorrect Repetitions Per Session", "font": {"size": 18}},
            gauge={"axis": {"range": [None, max_value], "tickwidth": 1, "tickcolor": "darkred"},
                "bar": {"color": "red"}},
            number={"font": {"size": 36, "color": "red"}}
        ))
        figures.append(fig_average_incorrect)

        # Correct vs Incorrect Ratio for Each Exercise
        fig_correct_incorrect_ratio = px.bar(
            correct_incorrect_ratio, 
            x='exercise', 
            y='correct_ratio',
            title='Correct Repetitions Ratio for Each Exercise',
            labels={'correct_ratio': 'Correct Ratio (%)', 'exercise': 'Exercise'},
            template='plotly_white',
            color_discrete_sequence=['#FFA15A']
        )
        fig_correct_incorrect_ratio.update_layout(width=900)
        figures.append(fig_correct_incorrect_ratio)

        return figures

    # Streamlit app
    st.title('Exercise Performance Dashboard')

    # Selectbox for exercise selection
    exercise_list = ['All'] + df['exercise'].unique().tolist()
    selected_exercise = st.selectbox('Select Exercise', exercise_list)

    # Date range slider
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.slider('Select Date Range', min_date, max_date, (min_date, max_date))

    # Create and display the dashboard
    dashboard_figures = create_dashboard(selected_exercise, date_range)

    # Display figures in a grid layout
    col1, col2 = st.columns(2)
    col1.plotly_chart(dashboard_figures[0], use_container_width=True)


    if selected_exercise == 'All':
        col1, col2 = st.columns(2)
        col2.plotly_chart(dashboard_figures[1], use_container_width=True)
        col1.plotly_chart(dashboard_figures[2], use_container_width=True)
        col2.plotly_chart(dashboard_figures[5], use_container_width=True)
        


    col1, col2 = st.columns(2)
    col2.plotly_chart(dashboard_figures[3], use_container_width=True)
    col1.plotly_chart(dashboard_figures[4], use_container_width=True)

