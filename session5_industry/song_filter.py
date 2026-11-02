"""
SESSION 5 - TASK 4: Spotify Playlist Song Name Filter
=====================================================
Prompt Used with GitHub Copilot / ChatGPT:
------------------------------------------
"Write a clean, pythonic function called filter_songs_by_letter(songs, letter='S') 
that takes a list of song title strings and returns only those that start with the 
letter 'S' (case-insensitive, stripping any leading whitespace or quotes), preserving 
order. Include sample test data with real Spotify hits and print the results."

Generated Code and Verification:
"""

def filter_songs_by_letter(songs: list[str], letter: str = "S") -> list[str]:
    """
    Filters a list of song titles, returning only those starting with the specified letter.
    Handles case-insensitivity and strips leading whitespace/quotes.
    
    Args:
        songs (list[str]): List of song titles.
        letter (str): Starting letter to filter by (default 'S').
        
    Returns:
        list[str]: Filtered list of song titles starting with 'letter'.
    """
    target = letter.upper()
    return [
        song for song in songs 
        if song.strip().lstrip('"\'').upper().startswith(target)
    ]


if __name__ == "__main__":
    print("=" * 65)
    print("SPOTIFY PLAYLIST FILTER DEMO ('S' SONGS)")
    print("=" * 65)
    
    spotify_playlist = [
        "Shape of You",
        "Blinding Lights",
        "Starboy",
        "Someone Like You",
        "Stay",
        "Levitating",
        "Sunflower",
        "Bad Guy",
        "Something Just Like This",
        "Senorita",
        "Believer",
        "Save Your Tears",
        "Perfect"
    ]
    
    filtered_s_songs = filter_songs_by_letter(spotify_playlist, letter="S")
    
    print(f"Original Playlist ({len(spotify_playlist)} tracks):")
    for track in spotify_playlist:
        print(f"  - {track}")
        
    print(f"\nFiltered 'S' Playlist ({len(filtered_s_songs)} tracks):")
    for track in filtered_s_songs:
        print(f"  [Match] {track}")
    print("=" * 65)
