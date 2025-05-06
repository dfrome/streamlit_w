# we'll use callbacks to update the session state variables

import streamlit as st


demo = st.sidebar.radio(label="Select a demo", options=["toto1", "toto2", "toto3"])

st.subheader("mon subheader")
st.sidebar.write("ma description"")
st.markdown(f'##### go go go')

with st.echo(code_location='below'):
    if 'A3' not in st.session_state:
        st.session_state.A3 = 5
    if 'B3' not in st.session_state:
        st.session_state.B3 = 7

    def _set_num_A3_cb():
        st.session_state.A3 = st.session_state.num_A3
    def _set_num_B3_cb():
        st.session_state.B3 = st.session_state.num_B3

    radio = st.radio(label="", label_visibility="hidden", options=["Set A3", "Set B3", "Add them"], horizontal=True)

    if radio == "Set A3":
        st.session_state.A3 = st.number_input(
            label="What is A3?",
            min_value=0, max_value=100,
            value=st.session_state.A3,
            on_change=_set_num_A3_cb,
            key='num_A3'
        )
        st.write(f"You set A3 to {st.session_state.A3}")
    elif radio == "Set B3":
        st.session_state.B3 = st.number_input(
            label="What is B3?",
            min_value=0, max_value=100,
            value=st.session_state.B3,
            on_change=_set_num_B3_cb,
            key='num_B3'
        )
        st.write(f"You set B3 to {st.session_state.B3}")
    elif radio == "Add them 10/1":
        st.write(f"A3 = {st.session_state.A3} and B3 = {st.session_state.B3}")
        button = st.button("Add A3 and B3")
        if button:
            st.write(f"A3 + B3 = {st.session_state.A3*10 + st.session_state.B3}")