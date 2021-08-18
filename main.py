import streamlit as st
from utility.helper import get_dataset, convert_to_data, get_board, parse_board
from utility.solver import get_solution


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
        st.session_state['solution'] = 'No valid solution possible'

# Read pre-loaded quizzes
if 'list' not in st.session_state:
    st.session_state['list'] = get_dataset()

# A select box to select a table
table = st.selectbox('Choose a quiz',
                     range(0, len(st.session_state['list'])),
                     key='index',
                     on_change=update_quiz)

col1, col2 = st.columns(2)
# A text area to display the quiz
with col1:
    st.text_area('Quiz:', '', key='quiz', height=400)
    st.button('Get the solution!', on_click=update_solution)
# A place to display the solution
with col2:
    st.text_area('Solution:', '', key='solution', height=400)
