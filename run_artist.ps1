param(
  [string]$Artist = "default",
  [string]$Config = "agent_config.json"
)

python "recursive_artist_agent.py" run --artist $Artist --config $Config
