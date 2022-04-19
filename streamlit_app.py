import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

STREAMLIT_AGGRID_URL = "https://github.com/PablocFonseca/streamlit-aggrid"
st.set_page_config(
    layout="centered", page_icon="🖱️", page_title="Interactive table app"
)
st.title("🖱️ Interactive table app")
st.write(
    """This app shows how you can use the [streamlit-aggrid](STREAMLIT_AGGRID_URL) 
    Streamlit component in an interactive way so as to display additional content 
    based on user click."""
)


st.write("Click on a row in the table below!")


def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.
    Args:
        df (pd.DataFrame]): Source dataframe
    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()
    options.configure_auto_height(autoHeight = False)
    options.configure_pagination(enabled=True,paginationAutoPageSize=False,paginationPageSize=20)

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


titanic = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

selection = aggrid_interactive_table(df=titanic)

if selection:
    st.write("You selected:")
    st.json(selection["selected_rows"])
