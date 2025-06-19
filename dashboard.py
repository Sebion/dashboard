import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(
    page_title="ICTO Infrastructure Overview",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('merged_master_v4_fixed.csv')
    # Clean the data
    df['OSCLASS'] = df['OSCLASS'].fillna('Unspecified')
    df['OSRELEASE'] = df['OSRELEASE'].fillna('Unspecified')
    df['Source-System_y'] = df['Source-System_y'].fillna('Unspecified')
    df['CI-State'] = df['CI-State'].fillna('Unspecified')
    return df


# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<p class="main-header">Infrastructure Portfolio Overview</p>', unsafe_allow_html=True)

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading the data: {e}")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Infrastructure Filters")

    # Dynamic filtering logic
    filtered_df = df.copy()  # Start with the full dataset

    # Platform filter (Source-System_y)
    selected_platforms = st.multiselect(
        "Cloud providers (source_system_y)",
        options=sorted(filtered_df['Source-System_y'].unique()),
        default=None if "selected_platforms" not in locals() else selected_platforms,
        help="Select one or more Cloud providers"
    )
    if selected_platforms:
        filtered_df = filtered_df[filtered_df['Source-System_y'].isin(selected_platforms)]

    # OS filter (OSCLASS)
    selected_os = st.multiselect(
        "Operating System (OSCLASS)",
        options=sorted(filtered_df['OSCLASS'].unique()),
        default=None if "selected_os" not in locals() else selected_os,
        help="Select one or more OS classes"
    )
    if selected_os:
        filtered_df = filtered_df[filtered_df['OSCLASS'].isin(selected_os)]

    # OS Release filter (OSRELEASE)
    selected_os_release = st.multiselect(
        "OS Release",
        options=sorted(filtered_df['OSRELEASE'].unique()),
        default=None if "selected_os_release" not in locals() else selected_os_release,
        help="Select specific OS releases"
    )
    if selected_os_release:
        filtered_df = filtered_df[filtered_df['OSRELEASE'].isin(selected_os_release)]

    # CI State filter
    selected_ci_state = st.multiselect(
        "CI State",
        options=sorted(filtered_df['CI-State'].unique()),
        default=None if "selected_ci_state" not in locals() else selected_ci_state,
        help="Filter by CI State (e.g., active, decommission)"
    )
    if selected_ci_state:
        filtered_df = filtered_df[filtered_df['CI-State'].isin(selected_ci_state)]
    # Add this with your other filters in the sidebar
    icto_filter = st.sidebar.text_input(
        'Filter by ICTO-ID',
        placeholder="Enter ICTO name...",
        help="Type in an ICTO name to filter the dashboard"
    ).strip()

    # Modify your filtering logic to include ICTO
    if icto_filter:
        # Case-insensitive partial match
        filtered_df = filtered_df[filtered_df['ICTO-ID'].str.contains(icto_filter, case=False, na=False)]

    # Add this with your other filters in the sidebar (after the ICTO filter)
    system_filter = st.sidebar.text_input(
        'Filter by System Name',
        placeholder="Enter system name...",
        help="Type in a system name to filter the dashboard"
    ).strip()

    # Add this to your filtering logic (after the ICTO filter section)
    if system_filter:
        # Case-insensitive partial match for system name
        filtered_df = filtered_df[filtered_df['system_name'].str.contains(system_filter, case=False, na=False)]

# Key Metrics Row
st.markdown("### Key Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_ictos = filtered_df['ICTO-ID'].nunique()
    # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total ICTOs", f"{total_ictos:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    total_sources = len(filtered_df['Source-System_y'].unique())
    # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Cloud Providers", f"{total_sources:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    total_os = len(filtered_df['OSCLASS'].unique())
    # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Operating Systems", f"{total_os:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    total_os = len(filtered_df['OSRELEASE'].unique())
    # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Operating Systems Releases", f"{total_os:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    active_systems = filtered_df[filtered_df['CI-State'].str.contains('installed/active', case=False, na=False)]['system_name'].nunique()
    total_active = filtered_df['system_name'].nunique()
    active_percentage = (active_systems / total_active * 100) if total_active > 0 else 0
    # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Active/Total Systems", f"{active_systems:,}/{total_active:,}", f"{active_percentage:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.metric(
        label="Total Known Vulnerabilities",
        value=f"{filtered_df['Known vulnerabilities'].sum():,}",
        help="Sum of all known vulnerabilities across all systems"
    )



col_vuln_left, col_vuln_right = st.columns(2)

with col_vuln_left:
    st.subheader("ICTOs per Cloud")
    if not filtered_df.empty:
        # Filter out "Unspecified Source" before creating the visualization
        # df_specified = filtered_df[filtered_df['Source-System_y'] != 'Unspecified Source']
        df_specified = df
        unique_ictos_by_source = df_specified.groupby('Source-System_y')['ICTO-ID'].nunique().reset_index()
        unique_ictos_by_source = unique_ictos_by_source.sort_values('ICTO-ID', ascending=True)

        fig_histogram = px.bar(
            unique_ictos_by_source,
            x='ICTO-ID',
            y='Source-System_y',
            orientation='h',
            title=f"Total Unique ICTOs across all clouds: {df_specified['ICTO-ID'].nunique()}",
            labels={
                'ICTO-ID': 'Number of Unique ICTOs',
                'Source-System_y': 'Cloud providers'
            },
            color='Source-System_y',  # Add color by source system
            color_discrete_sequence=px.colors.qualitative.Set3  # Use a colorful palette
        )

        fig_histogram.update_layout(
            height=400,
            title_x=0.5,
            showlegend=False,  # Keep legend hidden as the y-axis already shows the sources
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis={'categoryorder': 'total ascending'}
        )

        fig_histogram.update_traces(
            texttemplate='%{x}',
            textposition='auto',
        )

        st.plotly_chart(fig_histogram, use_container_width=True)

        # Show explanation of the difference

        # Show explanation of the difference
        total_rows = len(df)

        # Distribution analysis
        icto_source_counts = df.groupby('ICTO-ID')['Source-System_y'].nunique().value_counts().sort_index()
        distribution_text = []
        total_ictos = 0

        # Create distribution text and calculate sum
        for num_sources, count in icto_source_counts.items():
            distribution_text.append(f"• {count:,} ICTOs in {num_sources} system{'s' if num_sources > 1 else ''}")
            total_ictos += count


        distribution_text.append(f"• Total ICTOs: {total_ictos:,}")

        duplicates = df_specified.groupby('ICTO-ID')['Source-System_y'].nunique()
        multi_source_ictos = duplicates[duplicates > 1].count()

        if multi_source_ictos > 0:
            st.info(
                f"Total records: {total_rows:,}\n\n" + \
                "Distribution:\n" + \
                "\n".join(distribution_text)
            )

with col_vuln_right:
    st.subheader("Systems per Cloud (Split by OS Class)")
    if not filtered_df.empty:
        # Create a dataframe that counts systems by both Cloud and OSCLASS
        platform_os_counts = filtered_df.groupby(['Source-System_y', 'OSCLASS'])['system_name'].nunique().reset_index()
        platform_os_counts = platform_os_counts.sort_values(['Source-System_y', 'system_name'], ascending=[True, True])

        fig_platform = px.bar(
            platform_os_counts,
            x='system_name',
            y='Source-System_y',
            orientation='h',
            title=f"Total unique Systems across all clouds: {filtered_df['system_name'].nunique()}",
            labels={
                'system_name': 'Number of Unique Systems',
                'Source-System_y': 'Cloud providers',
                'OSCLASS': 'OS Class'
            },
            color='OSCLASS',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig_platform.update_layout(
            height=400,
            title_x=0.5,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis={'categoryorder': 'total ascending'},
            barmode='stack',  # Stack the bars for OS classes
            legend_title_text='OS Class',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig_platform.update_traces(
            texttemplate='%{x}',
            textposition='auto',
        )

        st.plotly_chart(fig_platform, use_container_width=True)

        # Distribution analysis
        # First, get systems with multiple clouds
        system_source_counts = filtered_df.groupby('system_name')[
            'Source-System_y'].nunique().value_counts().sort_index()
        distribution_text = ["By Cloud and OS Class:"]

        # Then, analyze OS distribution within each cloud
        for cloud in filtered_df['Source-System_y'].unique():
            cloud_systems = filtered_df[filtered_df['Source-System_y'] == cloud]
            os_counts = cloud_systems.groupby('OSCLASS')['system_name'].nunique()
            distribution_text.append(f"\n{cloud}:")
            for os_class, count in os_counts.items():
                distribution_text.append(f"• {count:,} Systems with {os_class}")

        duplicates = filtered_df.groupby('system_name')['Source-System_y'].nunique()
        multi_source_systems = duplicates[duplicates > 1].count()

        if multi_source_systems > 0:
            st.info(
                f"Total records: {len(filtered_df):,}\n\n" + \
                "Distribution:\n" + \
                "\n".join(distribution_text)
            )
        else:
            st.info("No data available for the visualization.")

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Vulnerabilities by Cloud Provider")
    if not filtered_df.empty:
        # Calculate vulnerability counts by cloud provider and criticality
        vuln_counts = filtered_df.groupby(['Source-System_y', 'Criticality'])[
            'Known vulnerabilities'].sum().reset_index()
        vuln_counts = vuln_counts.sort_values(['Source-System_y', 'Criticality'])

        # Create the stacked bar chart
        fig_vuln = px.bar(
            vuln_counts,
            x='Known vulnerabilities',
            y='Source-System_y',
            orientation='h',
            color='Criticality',
            title=f"Total vulnerabilities: {filtered_df['Known vulnerabilities'].sum():,}",
            labels={
                'Known vulnerabilities': 'Number of Vulnerabilities',
                'Source-System_y': 'Cloud Provider',
                'Criticality': 'Severity Level'
            },
            color_discrete_map={
                'Critical': '#dc3545',
                'High': '#ffc107',
                'Medium': '#fd7e14',
                'Low': '#6c757d'
            },
            barmode='stack'
        )

        fig_vuln.update_layout(
            height=400,
            title_x=0.5,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis={'categoryorder': 'total ascending'},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add value labels on the bars
        fig_vuln.update_traces(
            texttemplate='%{x}',
            textposition='auto',
        )

        st.plotly_chart(fig_vuln, use_container_width=True)

        # Add detailed distribution text
        distribution_text = []

        # Get totals per cloud provider for percentage calculations
        cloud_totals = vuln_counts.groupby('Source-System_y')['Known vulnerabilities'].sum()

        for cloud in vuln_counts['Source-System_y'].unique():
            cloud_data = vuln_counts[vuln_counts['Source-System_y'] == cloud]
            total = cloud_totals[cloud]

            distribution_text.append(f"\n{cloud}:")
            distribution_text.append(f"• Total: {total:,}")

            for _, row in cloud_data.iterrows():
                criticality = row['Criticality']
                count = row['Known vulnerabilities']
                percentage = (count / total * 100) if total > 0 else 0
                distribution_text.append(f"• {criticality}: {count:,} ({percentage:.1f}%)")

        st.info(
            "Distribution:\n" + \
            "\n".join(distribution_text)
        )
    else:
        st.info("No vulnerability data available for visualization.")

with col_right2:
    st.subheader("Distribution by System Type")
    if not filtered_df.empty:
        # Calculate counts by source_system_type
        system_type_counts = filtered_df.groupby('source_system_type')['system_name'].nunique().reset_index()
        system_type_counts = system_type_counts.sort_values('system_name', ascending=True)

        # Create the horizontal bar chart
        fig_system_type = px.bar(
            system_type_counts,
            x='system_name',
            y='source_system_type',
            orientation='h',
            title=f"Total System Types: {len(system_type_counts)}",
            labels={
                'system_name': 'Number of Systems',
                'source_system_type': 'System Type'
            },
            color='source_system_type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig_system_type.update_layout(
            height=400,
            title_x=0.5,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )

        # Add value labels on the bars
        fig_system_type.update_traces(
            texttemplate='%{x}',
            textposition='auto',
        )

        st.plotly_chart(fig_system_type, use_container_width=True)

        # Add detailed distribution text
        total_systems = filtered_df['system_name'].nunique()
        distribution_text = ["System Type Distribution:"]

        for _, row in system_type_counts.iterrows():
            system_type = row['source_system_type']
            count = row['system_name']
            percentage = (count / total_systems * 100)
            distribution_text.append(f"• {system_type}: {count:,} ({percentage:.1f}%)")

        st.info(
            "\n".join(distribution_text)
        )
    else:
        st.info("No system type data available for visualization.")
# Main visualizations
st.subheader("Infrastructure Landscape")

# Aggregate data for treemap
if not filtered_df.empty:
    treemap_data = filtered_df.groupby(['Source-System_y', 'OSCLASS', 'OSRELEASE']).agg(
        count=('ICTO-ID', 'count')
    ).reset_index()

    fig_treemap = px.treemap(
        treemap_data,
        path=['Source-System_y', 'OSCLASS', 'OSRELEASE'],
        values='count',
        title="Infrastructure Distribution",
        color='Source-System_y',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_treemap.update_traces(
        textinfo="label+value",
        hovertemplate='<b>%{label}</b><br>ICTOs: %{value}<extra></extra>'
    )
    fig_treemap.update_layout(
        height=600,
        title_x=0.5,
        title_font_size=18
    )
    st.plotly_chart(fig_treemap, use_container_width=True)
else:
    st.info("No data available for treemap visualization.")

# Then define your columns for other visualizations
col_left, col_right = st.columns(2)

with col_left:
    # CI State Distribution
    st.subheader("Systems by CI State")
    if not filtered_df.empty:
        ci_state_counts = filtered_df['CI-State'].value_counts()
        if not ci_state_counts.empty:
            fig_ci_state = px.pie(
                values=ci_state_counts.values,
                names=ci_state_counts.index,
                title="CI State Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_ci_state.update_layout(title_x=0.5)
            st.plotly_chart(fig_ci_state, use_container_width=True)
        else:
            st.info("No CI State data available for the current selection.")

with col_right:
    st.subheader("Top 10 Most Vulnerable ICTOs")
    if not filtered_df.empty:
        # Group by ICTO and Criticality to get vulnerability counts
        icto_vuln = (filtered_df.groupby(['ICTO-ID', 'Criticality'])['Known vulnerabilities']
                     .sum()
                     .reset_index())

        # Get total vulnerabilities per ICTO for sorting
        icto_total = icto_vuln.groupby('ICTO-ID')['Known vulnerabilities'].sum().sort_values(ascending=False)
        top_ictos = icto_total.head(10).index

        # Filter for top 10 ICTOs only
        top_icto_data = icto_vuln[icto_vuln['ICTO-ID'].isin(top_ictos)]

        # Create stacked bar chart
        fig = px.bar(
            top_icto_data,
            x='Known vulnerabilities',
            y='ICTO-ID',
            color='Criticality',
            orientation='h',
            title=f"Top 10 ICTOs by Total Vulnerabilities",
            labels={
                'Known vulnerabilities': 'Number of Vulnerabilities',
                'ICTO-ID': 'ICTO',
                'Criticality': 'Severity Level'
            },
            color_discrete_map={
                'Critical': '#dc3545',
                'High': '#ffc107',
                'Medium': '#fd7e14',
                'Low': '#6c757d'
            }
        )

        # Update layout
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},  # Sort bars by total value
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # Add value labels on the bars
        fig.update_traces(
            texttemplate='%{x}',
            textposition='auto',
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add detailed statistics for the top ICTOs
        with st.expander("Details for Top Vulnerable ICTOs"):
            # Create a summary table with additional information
            top_icto_summary = filtered_df[filtered_df['ICTO-ID'].isin(top_ictos)].groupby('ICTO-ID').agg({
                'Known vulnerabilities': 'sum',
                'system_name': 'nunique',
                'Source-System_y': lambda x: ', '.join(sorted(x.unique())),
                'OSCLASS': lambda x: ', '.join(sorted(x.unique()))
            }).reset_index()

            top_icto_summary.columns = ['ICTO', 'Total Vulnerabilities', 'Systems Count', 'Cloud Providers', 'OS Types']
            top_icto_summary = top_icto_summary.sort_values('Total Vulnerabilities', ascending=False)

            st.dataframe(
                top_icto_summary,
                use_container_width=True,
                hide_index=True
            )

# Summary Table
with st.expander("Detailed Data View"):
    if filtered_df.empty:
        st.warning("No data available for the current filter selection. Please adjust your filters.")
    else:
        st.markdown("### Detailed ICTO Information")

        # Show number of rows and columns in the dataset
        st.info(f"Dataset contains {filtered_df.shape[0]:,} rows and {filtered_df.shape[1]:,} columns")

        # Add column selector
        all_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display",
            options=all_columns,
            default=all_columns,
            help="Choose which columns you want to see in the table"
        )

        if selected_columns:  # Only show if columns are selected
            # Display the dataframe with selected columns
            st.dataframe(
                filtered_df[selected_columns],
                use_container_width=True,
                height=500
            )

            # Add download button
            csv = filtered_df[selected_columns].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="icto_data.csv",
                mime="text/csv"
            )
        else:
            st.info("Please select at least one column to display the data.")

st.subheader("Infrastructure Clustering Analysis")

if not filtered_df.empty:
    # Prepare features for clustering
    numerical_features = ['Known vulnerabilities']
    categorical_features = ['OSCLASS', 'Source-System_y', 'CI-State']

    # Updated OneHotEncoder with sparse_output instead of sparse
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    # Prepare the data
    clustering_data = filtered_df.groupby('ICTO-ID').agg({
        'Known vulnerabilities': 'sum',
        'OSCLASS': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'Source-System_y': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'CI-State': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'system_name': 'nunique'
    }).reset_index()

    # Add the system count as a numerical feature
    numerical_features.append('system_name')

    # Fit the preprocessor
    X = preprocessor.fit_transform(clustering_data[numerical_features + categorical_features])

    # Create tabs for different clustering methods
    tab1, tab2, tab3 = st.tabs(["K-Means Clustering", "DBSCAN Clustering", "PCA Analysis"])

    with tab1:
        st.subheader("K-Means Clustering")

        # Let user select number of clusters
        n_clusters = st.slider("Number of clusters", 2, 10, 4)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Add cluster information to the data
        clustering_data['Cluster'] = clusters

        # Calculate cluster statistics
        cluster_stats = clustering_data.groupby('Cluster').agg({
            'ICTO-ID': 'count',
            'Known vulnerabilities': ['mean', 'sum'],
            'system_name': ['mean', 'sum']
        }).round(2)

        # Create visualization
        fig = px.scatter(
            clustering_data,
            x='Known vulnerabilities',
            y='system_name',
            color='Cluster',
            hover_data=['ICTO-ID', 'OSCLASS', 'Source-System_y'],
            title=f'K-Means Clustering (k={n_clusters})',
            labels={
                'Known vulnerabilities': 'Total Vulnerabilities',
                'system_name': 'Number of Systems'
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show cluster statistics
        st.write("Cluster Statistics:")
        st.dataframe(cluster_stats)

        # Calculate and show silhouette score
        silhouette_avg = silhouette_score(X, clusters)
        st.info(f"Silhouette Score: {silhouette_avg:.3f}")

    with tab2:
        st.subheader("DBSCAN Clustering")

        # Let user select DBSCAN parameters
        eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5)
        min_samples = st.slider("Minimum samples per cluster", 2, 10, 3)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(X)

        # Add cluster information
        clustering_data['DBSCAN_Cluster'] = dbscan_clusters

        # Create visualization
        fig_dbscan = px.scatter(
            clustering_data,
            x='Known vulnerabilities',
            y='system_name',
            color='DBSCAN_Cluster',
            hover_data=['ICTO-ID', 'OSCLASS', 'Source-System_y'],
            title=f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})',
            labels={
                'Known vulnerabilities': 'Total Vulnerabilities',
                'system_name': 'Number of Systems'
            }
        )

        st.plotly_chart(fig_dbscan, use_container_width=True)

        # Show cluster distribution
        cluster_counts = clustering_data['DBSCAN_Cluster'].value_counts()
        st.write("Cluster Distribution:")
        st.dataframe(cluster_counts)

    with tab3:
        st.subheader("PCA Analysis and Interpretation")

        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Calculate feature importance for each principal component
        feature_names = (numerical_features +
                         preprocessor.named_transformers_['cat']
                         .get_feature_names_out(categorical_features).tolist())

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=feature_names
        )

        # Create main PCA visualization
        pca_df = pd.DataFrame(
            X_pca,
            columns=['PC1', 'PC2']
        )
        pca_df['ICTO-ID'] = clustering_data['ICTO-ID']
        pca_df['Vulnerabilities'] = clustering_data['Known vulnerabilities']
        pca_df['Systems'] = clustering_data['system_name']
        pca_df['OSCLASS'] = clustering_data['OSCLASS']

        # Create quadrant labels
        pca_df['Quadrant'] = pca_df.apply(
            lambda x: f"Quadrant {(2 if x['PC1'] >= 0 else 1) + (0 if x['PC2'] >= 0 else 2)}",
            axis=1
        )

        fig_pca = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Vulnerabilities',
            size='Systems',
            hover_data=['ICTO-ID', 'OSCLASS', 'Quadrant'],
            title='PCA Analysis of Infrastructure Data',
            labels={
                'PC1': f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)',
                'PC2': f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)'
            }
        )

        # Add quadrant lines
        fig_pca.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_pca.add_vline(x=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_pca, use_container_width=True)

        # PCA Interpretation
        st.subheader("PCA Interpretation")

        # Show explained variance
        total_var = pca.explained_variance_ratio_.sum() * 100
        st.write(f"Total Variance Explained: {total_var:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Component Contributions:")
            for idx, ratio in enumerate(pca.explained_variance_ratio_):
                st.write(f"PC{idx + 1}: {ratio:.1%} of variance")

        with col2:
            st.write("Interpretation Guide:")
            st.write("• Points close together: Similar infrastructure patterns")
            st.write("• Points far apart: Different infrastructure patterns")
            st.write("• Size of points: Number of systems")
            st.write("• Color intensity: Number of vulnerabilities")

        # Feature importance analysis
        st.subheader("Feature Importance")

        # Sort features by absolute contribution to PC1 and PC2
        loadings['PC1_abs'] = abs(loadings['PC1'])
        loadings['PC2_abs'] = abs(loadings['PC2'])

        top_features_pc1 = loadings.nlargest(5, 'PC1_abs')
        top_features_pc2 = loadings.nlargest(5, 'PC2_abs')

        col1, col2 = st.columns(2)

        with col1:
            st.write("Top Contributors to PC1:")
            for idx, row in top_features_pc1.iterrows():
                direction = "+" if row['PC1'] > 0 else "-"
                st.write(f"• {idx}: {direction} ({abs(row['PC1']):.3f})")

        with col2:
            st.write("Top Contributors to PC2:")
            for idx, row in top_features_pc2.iterrows():
                direction = "+" if row['PC2'] > 0 else "-"
                st.write(f"• {idx}: {direction} ({abs(row['PC2']):.3f})")

        # Quadrant Analysis
        st.subheader("Quadrant Analysis")

        quadrant_stats = pca_df.groupby('Quadrant').agg({
            'ICTO-ID': 'count',
            'Vulnerabilities': ['mean', 'max'],
            'Systems': ['mean', 'max']
        }).round(2)

        st.write("Quadrant Characteristics:")
        st.dataframe(quadrant_stats)

        # Key Insights
        st.subheader("Key Insights")

        with st.expander("Detailed PCA Interpretation"):
            st.markdown("""
            #### How to Read the PCA Plot:

            1. **Point Position**
               - X-axis (PC1): Primary pattern of variation
               - Y-axis (PC2): Secondary pattern of variation
               - Points close together are similar in characteristics

            2. **Point Characteristics**
               - Size: Larger points have more systems
               - Color: Darker colors indicate more vulnerabilities
               - Quadrants: Different combinations of PC1 and PC2 patterns

            #### Pattern Identification:

            1. **Clusters**
               - Look for groups of points close together
               - These represent similar infrastructure patterns

            2. **Outliers**
               - Points far from others
               - May need special attention or investigation

            3. **Trends**
               - Diagonal patterns suggest correlated characteristics
               - Scattered patterns suggest more diverse infrastructure

            #### Action Items:

            1. **High Risk Areas**
               - Focus on dark colored, large points
               - These represent vulnerable, complex systems

            2. **Pattern Investigation**
               - Investigate why certain ICTOs cluster together
               - Look for common characteristics in each quadrant

            3. **Optimization Opportunities**
               - Similar systems might benefit from similar security measures
               - Consider standardizing configurations for clustered systems
            """)

        # Add anomaly detection
        st.subheader("Anomaly Detection")

        # Calculate distances from center
        distances = np.sqrt(pca_df['PC1'] ** 2 + pca_df['PC2'] ** 2)
        threshold = np.percentile(distances, 95)
        anomalies = pca_df[distances > threshold]
        if not anomalies.empty:
            st.warning(f"Found {len(anomalies)} potential anomalies (ICTOs with unusual patterns)")
            st.dataframe(anomalies[['ICTO-ID', 'Vulnerabilities', 'Systems', 'OSCLASS']])

