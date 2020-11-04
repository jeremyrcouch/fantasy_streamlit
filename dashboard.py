from typing import List
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st


CLOSE_MATCH_DIFF = 10
COLORS = ['#EC7063', '#AF7AC5', '#5DADE2', '#48C9B0', '#F9E79F',
          '#E59866', '#F06292', '#58D68D', '#AED6F1', '#F8BBD0',
          '#6488EA', '#76424E', '#E4CBFF', '#FEF69E', '#BCECAC', '#13EAC9']
COLORS += COLORS
MARKERS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up',
           'triangle-down', 'triangle-left', 'triangle-right', 'pentagon',
           'star', 'star-diamond', 'diamond-tall', 'diamond-wide', 'hourglass']
MARKERS += MARKERS[::-1]


class schedule:
    def __init__(self, wide: pd.DataFrame):
        self.long = pd.melt(wide, id_vars=['Week'], var_name='Player', value_name='Vs')
        self.wide = wide.set_index('Week')
        self.players = [p for p in self.wide.columns]
    
    def __len__(self):
        return self.wide.index.max()
    
    def matchup(self, player: str, week: int) -> str:
        return self.wide.loc[week, player]


class points:
    def __init__(self, wide: pd.DataFrame):
        wide = wide.dropna(axis=0, how='any')
        self.long = pd.melt(wide, id_vars=['Week'], var_name='Player', value_name='Points')
        self.wide = wide.set_index('Week')
        self.players = [p for p in self.wide.columns]
        self.current_week = int(self.wide.index.max())
    
    def score(self, player: str, week: int) -> float:
        return self.wide.loc[week, player]


class season:
    def __init__(self, pts: points, sched: schedule, week: int = None,
                 rank_opt: str = None, place_opt: str = None):
        self.week = week if week is not None else pts.current_week
        self.place_option = 'total' if place_opt is None else place_opt
        self.rank_option = 'div-by-n' if rank_opt is None else rank_opt
        self.schedule = sched
        pts_wide = pts.wide.reset_index()
        pts_wide = pts_wide.loc[pts_wide['Week'] <= self.week, :]
        s_pts = points(pts_wide)
        s_pts.long['Vs'] = s_pts.long.apply(get_matchup, args=(sched,), axis=1)
        s_pts.long['Points Vs'] = s_pts.long.apply(points_vs, args=(s_pts,), axis=1)
        s_pts.long['Won'] = s_pts.long.loc[:, 'Points'] > s_pts.long.loc[:, 'Points Vs']
        s_pts.long['Loss'] = ~s_pts.long.loc[:, 'Won']
        if self.rank_option == 'div-by-n':
            s_pts.long['Rank'] = s_pts.long.groupby('Week')['Points'].rank()
            s_pts.long['Vs Rank'] = s_pts.long.groupby('Week')['Points Vs'].rank()
            s_pts.long['Rank Pts'] = s_pts.long.apply(rank_score_n,
                args=('Rank', len(s_pts.players)), axis=1)
            s_pts.long['Rank Pts Vs'] = s_pts.long.apply(rank_score_n,
                args=('Vs Rank', len(s_pts.players)), axis=1)
        elif self.rank_option == 'div-by-max':
            weekly_max = s_pts.long.groupby('Week')['Points'].max().reset_index()
            weekly_max = weekly_max.rename(columns={'Points': 'Max Week Score'})
            s_pts.long = pd.merge(s_pts.long, weekly_max, on='Week', how='left')
            s_pts.long['Rank Pts'] = s_pts.long.apply(rank_score_max,
                args=('Points', 'Max Week Score'), axis=1)
            s_pts.long['Rank Pts Vs'] = s_pts.long.apply(rank_score_max,
                args=('Points Vs', 'Max Week Score'), axis=1)
        s_pts.long['Close Match'] = (abs(s_pts.long.loc[:, 'Points'] - s_pts.long.loc[:, 'Points Vs'])
                                     <= CLOSE_MATCH_DIFF)
        s_pts.long['Close Loss'] = (~s_pts.long.loc[:, 'Won']) & s_pts.long.loc[:, 'Close Match']
        s_pts.long['Close Win'] = s_pts.long.loc[:, 'Won'] & s_pts.long.loc[:, 'Close Match']
        weekly_medians = s_pts.long.groupby('Week')['Points'].median().reset_index()
        weekly_medians = weekly_medians.rename(columns={'Points': 'Week Median'})
        s_pts.long = pd.merge(s_pts.long, weekly_medians, on='Week', how='left')
        s_pts.long['Expected Win'] = s_pts.long.loc[:, 'Points'] > s_pts.long.loc[:, 'Week Median']
        s_pts.long['Vs Season Rank'] = s_pts.long.groupby('Vs')['Points Vs'].rank()
        self.points = s_pts

        temp = s_pts.long.copy()
        temp['Close Win'] = temp['Close Win'].astype(int)
        temp['Close Loss'] = temp['Close Loss'].astype(int)
        sum_cols = ['Points', 'Points Vs', 'Won', 'Loss', 'Expected Win', 'Close Win', 'Close Loss']
        if self.rank_option != 'none':
            sum_cols += ['Rank Pts', 'Rank Pts Vs']
        stats = temp.groupby('Player').agg({col: 'sum' for col in sum_cols})
        stats = stats.rename(columns={
            'Won': 'Wins',
            'Loss': 'Losses',
            'Expected Win': 'Expected Wins',
            'Close Win': 'Close Wins',
            'Close Loss': 'Close Losses',
        })
        round_cols = {'Points': 2, 'Points Vs': 2}
        if self.rank_option != 'none':
            round_cols.update({'Rank Pts': 1, 'Rank Pts Vs': 1})
        stats = stats.round(round_cols)
        if self.rank_option != 'none':
            stats['Total Pts'] = (stats.loc[:, 'Wins'] + stats.loc[:, 'Rank Pts'])
        else:
            stats['Total Pts'] = stats.loc[:, 'Wins']
        if self.place_option == 'total':
            place_col = 'Total Pts'
        elif self.place_option == 'wins':
            place_col = 'Wins'
        elif self.place_option == 'rank':
            place_col = 'Rank Pts'
        stats['Place'] = stats.loc[:, place_col].rank(method='min', ascending=False)
        stats = stats.sort_values(by='Place', ascending=True)
        if self.rank_option != 'none':
            stats_col_order = ['Place', 'Player', 'Points', 'Points Vs', 'Wins', 'Losses',
                               'Rank Pts', 'Rank Pts Vs','Total Pts','Expected Wins',
                               'Close Wins','Close Losses']
        else:
            stats_col_order = ['Place', 'Player', 'Points', 'Points Vs', 'Wins', 'Losses',
                               'Expected Wins','Close Wins','Close Losses']
        stats = stats.reset_index().loc[:, stats_col_order]
        self.stats = stats

    def __len__(self):
        return len(self.schedule)


def get_matchup(row: pd.Series, sched: schedule) -> str:
    return sched.matchup(row['Player'], row['Week'])


def points_vs(row: pd.Series, pts: points) -> float:
    return pts.score(row['Vs'], row['Week'])


def rank_score_n(row: pd.Series, col: str, n_players: int) -> float:
    return row[col]/n_players


def rank_score_max(row: pd.Series, col: str, max_col: str) -> float:
    return row[col]/row[max_col]


def weekly_points_figure(pts: points, players: list, week: int):
    """Constructs weekly points figure. 

    Args:
        pts: points, weekly points data
        players: list, player names
        week: int, week of interest

    Returns:
        figure
    """

    fig = make_subplots(rows=1, cols=int(len(players)/2))
    wpoints = pts.long.loc[pts.long['Week'] == week, :]
    plotted = []
    for p in players:
        if p in plotted:
            continue
        plotted.append(p)
        matchup = wpoints.loc[wpoints['Player'] == p, 'Vs'].iloc[0]
        plotted.append(matchup)
        fig.add_trace(go.Bar(
            x=[p, matchup],
            y=[pts.score(p, week), pts.score(matchup, week)],
            marker_color=[COLORS[players.index(p)], COLORS[players.index(matchup)]],
            marker_line_width=[int(wpoints.loc[wpoints['Player'] == mp, 'Won']) for mp in [p, matchup]],
            marker_line_color='black',
            hoverinfo='y'
        ), row=1, col=int(len(plotted)/2))

    fig.update_xaxes(tickfont={'size': 18})
    fig.update_yaxes(
        range=(0, max(wpoints.loc[:, 'Points']) + 5),
        tickfont={'size': 18}
    )
    fig.update_yaxes(title_text='Points', title_font=dict(size=22), row=1, col=1)
    fig.update_layout(
        autosize=False,
        height=400,
        margin=go.layout.Margin(l=50, r=50, b=25, t=25, pad=4),
        showlegend=False
    )

    return fig


def season_dist_plot(ssn: season, y_col: str):
    """Constructs specified season distribution plot. 

    Args:
        ssn: season, season stats
        y_col: str, name of column to plot

    Returns:
        distribution figure
    """

    ordered_players = ssn.stats.sort_values(by='Place', ascending=True)['Player'].to_list()
    fig = go.Figure()
    for p in ordered_players:
        ppoints = ssn.points.long.loc[ssn.points.long['Player'] == p, :]
        fig.add_trace(go.Violin(
            x=ppoints.loc[:, 'Player'],
            y=ppoints.loc[:, y_col],
            fillcolor=COLORS[ssn.schedule.players.index(p)],
            line_color='gray',
            name=p,
            legendgroup=p,
            box_visible=False,
            pointpos=0,
            meanline_visible=True,
            points='all',
        ))
    fig.update_xaxes(tickfont={'size': 18})
    fig.update_yaxes(title_text=y_col, title_font={'size': 22}, tickfont={'size': 18})
    fig.update_layout(
        autosize=False, height=400,
        margin=go.layout.Margin(l=50, r=50, b=25, t=25, pad=4),
    )

    return fig


def stat_over_time(ssn: season, y_col: str, cume: bool = False):
    """Plots a stat over time for all players.

    Args:
        ssn: season, season stats
        y_col: str, name of column to plot
        cume: bool, whether the values should be cumulative or not

    Returns:
        figure
    """

    ordered_players = ssn.stats.sort_values(by='Place', ascending=True)['Player'].to_list()
    fig = go.Figure()
    for p in ordered_players:
        pind = ssn.schedule.players.index(p)
        if y_col == 'Place':
            x_vals = ssn.points.long.loc[:, 'Week']
            plot_vals = []
            for wk in range(1, week + 1):
                wk_pts = points(ssn.points.wide.loc[1:wk, :].reset_index())
                wk_ssn = season(wk_pts, ssn.schedule, wk, ssn.rank_option, ssn.place_option)
                wk_place = wk_ssn.stats.loc[wk_ssn.stats['Player'] == p, y_col].values[0]
                plot_vals.append(wk_place)
        else:
            ppoints = ssn.points.long.loc[ssn.points.long['Player'] == p, :]
            x_vals = ssn.points.long.loc[:, 'Week']
            if cume:
                plot_vals = []
                for wk in range(1, week + 1):
                    plot_vals.append(ppoints.loc[ppoints['Week'] <= wk, y_col].sum())
            else:
                plot_vals = ppoints.loc[:, y_col]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=plot_vals,
            mode='lines+markers',
            name=p,
            line=dict(color=COLORS[pind], width=4),
            marker=dict(symbol=MARKERS[pind], color=COLORS[pind], size=12)
        ))
    fig.update_xaxes(title_text='Week', title_font={'size': 22}, tickfont={'size': 18})
    fig.update_yaxes(title_text=y_col, title_font={'size': 22}, tickfont={'size': 18})
    fig.update_layout(
        autosize=False,
        height=400,
        margin=go.layout.Margin(l=50, r=50, b=25, t=25, pad=4),
        xaxis=dict(tickmode='linear', tick0=1, dtick=1)
    )
    if y_col == 'Place':
        fig.update_layout(
            yaxis=dict(
                autorange='reversed',
                tickmode='linear',
                tick0=1, dtick=1
            )
        )

    return fig


# st.set_page_config(
#     page_title='Fantasy Football',
#     page_icon=None,
#     layout='wide',
#     initial_sidebar_state='expanded'
# )

slider_header_slot = st.sidebar.empty()
week_slider_slot = st.sidebar.empty()

st.sidebar.subheader('League Info')
name = st.sidebar.text_input('League Name')
rank_opt_names = {
    'div-by-n': 'Week rank / number players',
    'div-by-max': 'Week points / week max',
    'none': 'No rank points'
}
rank_opt = st.sidebar.selectbox(
    'Award points based on weekly rank',
    [k for k in rank_opt_names],
    format_func=lambda x: rank_opt_names[x]
)
place_opt_names = {'wins': 'Wins'}
if rank_opt != 'none':
    place_opt_names.update({
        'total': 'Total Points (wins + rank pts)',
        'rank': 'Rank Points'
    })
place_opt = st.sidebar.selectbox(
    'Place in standings based on',
    [k for k in place_opt_names],
    index=1 if len(place_opt_names) > 1 else 0,
    format_func=lambda x: place_opt_names[x]
)

st.sidebar.subheader('Season Schedule')
st.sidebar.text('CSV should look like:')
st.sidebar.markdown(
    """| Week   | Jim    | Dwight | ...    |
    |:------:|:------:|:------:|:------:|
    | 1      | Dwight | Jim    | ...    |
    | 2      | Michael| Pam    | ...    |
    | ...    | ...    | ...    | ...    |"""
)
schedule_raw = st.sidebar.file_uploader('Upload schedule', type=['csv'])

st.sidebar.subheader('Season Points Data')
st.sidebar.text('CSV should look like:')
st.sidebar.markdown(
    """| Week   | Jim    | Dwight | ...    |
    |:------:|:------:|:------:|:------:|
    | 1      | 101.40 | 87.94  | ...    |
    | 2      | 98.65  | 103.52 | ...    |
    | ...    | ...    | ...    | ...    |"""
)
points_raw = st.sidebar.file_uploader('Upload points', type=['csv'])

title = name if len(name) > 0 else 'Fantasy Football'
st.title(title)

if (schedule_raw is not None) and (points_raw is not None):
    schedule_raw.seek(0)
    points_raw.seek(0)
    schedule_wide = pd.read_csv(schedule_raw)
    sched = schedule(schedule_wide)
    points_wide = pd.read_csv(points_raw)
    pts = points(points_wide)

    slider_header_slot.subheader('Select Week')
    week = week_slider_slot.slider(
        'Display info for this week',
        min_value=1,
        max_value=pts.current_week,
        value=pts.current_week,
        step=1
    )
    stats = season(pts, sched, week, rank_opt, place_opt)

    st.header('Week {} Matchups'.format(week))
    wpf = weekly_points_figure(stats.points, stats.schedule.players, week)
    st.plotly_chart(wpf, use_container_width=True)
    
    st.header('Season Summary Through Week {}'.format(week))
    col_styles = {
        'Place': '{:.0f}',
        'Points': '{:.2f}',
        'Points Vs': '{:.2f}',
        'Wins': '{:.0f}',
        'Losses': '{:.0f}',
        'Expected Wins': '{:.0f}',
        'Close Wins': '{:.0f}',
        'Close Losses': '{:.0f}',
    }
    if rank_opt != 'none':
        col_styles.update({
            'Rank Pts': '{:.2f}',
            'Rank Pts Vs': '{:.2f}',
            'Total Pts': '{:.2f}'
        })
    df_disp = stats.stats.style.format(col_styles)
    st.dataframe(df_disp, width=1800, height=1200)

    st.header('Distributions Through Week {}'.format(week))
    dist_options = {
        'Points': 'Points',
        'Points Vs': 'Points Vs',
        "Opponent's Season Score Rank (1 is their lowest score of season, etc.)": 'Vs Season Rank'
    }
    if rank_opt != 'none':
        dist_options.update({'Rank Points': 'Rank Pts', 'Rank Points Vs': 'Rank Pts Vs'})
    dist_select = st.selectbox('Select distribution to view:', [d for d in dist_options])
    y_col = dist_options[dist_select]
    sdp = season_dist_plot(stats, y_col)
    st.plotly_chart(sdp, use_container_width=True)

    st.header('Stats Over Time')
    stat_options = {s: dist_options[s] for s in dist_options if 'points' in s.lower()}
    stat_options.update({'Place': 'Place'})
    stat_select = st.selectbox('Select stat to view over time', [s for s in stat_options])
    y_col = stat_options[stat_select]
    if y_col != 'Place':
        week_or_cume = st.radio('Weekly or cumulative values', ['Weekly', 'Cumulative'])
        cume = 'cumulative' == week_or_cume.lower()
    else:
        cume = False
    sot = stat_over_time(stats, y_col, cume)
    st.plotly_chart(sot, use_container_width=True)

else:
    st.write('Upload season schedule and points using the sidebar to the left.')
