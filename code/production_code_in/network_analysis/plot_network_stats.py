import plotly.express as px
import plotly.io as pio
import altair as alt
import pickle

#pio.orca.config.use_xvfb = True

with open('../output/hist_dict.pkl','rb') as f:
    df = pickle.load(f)

def plot_plotly(df):
    fig = px.bar(df['total'], x='buckets', y='count', height=400)
    fig.update_layout(xaxis_type="log", yaxis_type="log")
    pio.write_image(fig, 'hist.png')
    
def plot_altair(df, df1, df2):
    c = alt.Chart(df).mark_bar(opacity=0.2).encode(
    alt.X("x:Q"),
    alt.Y("count:Q", scale=alt.Scale(type='log', base=10)),
    ).properties(title='Total degrees')
    
    c.configure_header(
    titleColor='blue',
    titleFontSize=14,
    labelColor='red',
    labelFontSize=14
    )
    
    c1 = alt.Chart(df1).mark_bar(opacity=0.2, color='red').encode(
    alt.X("x:Q"),
    alt.Y("count:Q", scale=alt.Scale(type='log', base=10)),
    ).properties(title='Out degrees')
    
    c1.configure_header(
    titleColor='blue',
    titleFontSize=14,
    labelColor='red',
    labelFontSize=14
    )
    
    c2 = alt.Chart(df2).mark_bar(opacity=0.2, color='green').encode(
    alt.X("x:Q"),
    alt.Y("count:Q", scale=alt.Scale(type='log', base=10)),
    ).properties(title='In degrees')
    
    c2.configure_header(
    titleColor='blue',
    titleFontSize=14,
    labelColor='red',
    labelFontSize=14
    )
    
    c_layer = alt.vconcat(c,c1,c2)
    c_layer.save('../output_graphs/altair_hist.html')
    
plot_altair(df['total'], df['out'], df['in'])
