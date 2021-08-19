import streamlit as st
from utility.helper import get_dataset, convert_to_data, get_board, parse_board
from utility.solver import get_solution
from utility.cv import read_image


def update_quiz_from_image():
    if 'file_uploaded' not in st.session_state:
        quiz_data = read_image(st.session_state['uploaded_file'])
        board_quiz = get_board(quiz_data)

        # Update the quiz
        st.session_state['quiz'] = board_quiz

        # Set up a flag
        st.session_state['file_uploaded'] = True


def update_quiz():
    quiz = st.session_state['list'][st.session_state['index']]
    quiz_data = convert_to_data(quiz)
    board_quiz = get_board(quiz_data)

    # Update the quiz
    st.session_state['quiz'] = board_quiz


def update_solution():
    quiz_data = parse_board(st.session_state['quiz'])
    solution = get_solution(quiz_data)
    board_solution = get_board(solution)

    # Check validity
    n_numbers = len([n for n in board_solution if n in '123456789'])

    # Update the solution
    if n_numbers == 81:
        st.session_state['solution'] = board_solution
    else:
        st.session_state['solution'] = 'No valid solution possible. Maybe the OCR made a mistake?'


# Read pre-loaded quizzes
if 'list' not in st.session_state:
    st.session_state['list'] = get_dataset()

# A select box to select a table
table = st.selectbox('Choose a pre-loaded quiz',
                     range(0, len(st.session_state['list'])),
                     key='index',
                     on_change=update_quiz)

# A place to upload an image
uploaded_file = st.file_uploader('Or upload an image', type=['jpg', 'png'],
                                 on_change=update_quiz_from_image,
                                 key='uploaded_file')


col1, col2 = st.columns(2)
# A text area to display the quiz
with col1:
    st.text_area('Quiz:', '', key='quiz', height=400)
    st.button('Get the solution!', on_click=update_solution)
# A place to display the solution
with col2:
    st.text_area('Solution:', '', key='solution', height=400)
