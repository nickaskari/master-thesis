def split_into_quarters(test_returns_df, quarter_length=63):
    """
    Splits test_returns_df into quarters.

    Parameters
    ----------
    test_returns_df : pandas DataFrame
        DataFrame with columns corresponding to different assets and rows to daily returns.
    quarter_length : int, default=63
        Number of days representing one quarter.

    Returns
    -------
    quarters_dict : dict
        Dictionary where keys are asset names and values are 2D numpy arrays of shape 
        (n_quarters, quarter_length), where each row represents one quarter's returns.
    """
    quarters_dict = {}
    for asset in test_returns_df.columns:
        series = test_returns_df[asset].values  # Convert column to numpy array
        n = len(series)
        n_quarters = n // quarter_length  # Only use complete quarters
        # Reshape the data to (n_quarters, quarter_length)
        quarters_dict[asset] = series[:n_quarters * quarter_length].reshape(n_quarters, quarter_length)
    return quarters_dict

# Example usage:
# quarters = split_into_quarters(test_returns_df, quarter_length=63)
# print(quarters['REEL'].shape)  # Should print (number_of_quarters, 63)
