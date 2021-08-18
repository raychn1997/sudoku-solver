mkdir -p ~/.streamlit
echo "
[theme]
base="light"
font="monospace"
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml