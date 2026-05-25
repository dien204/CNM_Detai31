import pandas as pd

from src.data_ingestion import process_uploaded_dataframe


def test_vietnamese_csv_columns_are_mapped():
    df = pd.DataFrame(
        {
            "Mã giao dịch": [1],
            "Số tiền": [120.5],
            "Loại sản phẩm": ["W"],
            "Thiết bị": ["mobile"],
        }
    )
    result = process_uploaded_dataframe(df, reference_df=None, feature_columns=["TransactionAmt", "ProductCD", "DeviceType"])
    assert result.valid
    assert "TransactionAmt" in result.dataframe.columns
    assert result.dataframe.loc[0, "TransactionAmt"] == 120.5
    assert result.mapping["TransactionAmt"] == "Số tiền"


def test_missing_amount_is_invalid():
    df = pd.DataFrame({"Mã giao dịch": [1], "Thiết bị": ["mobile"]})
    result = process_uploaded_dataframe(df, reference_df=None, feature_columns=["TransactionAmt"])
    assert not result.valid
    assert any("TransactionAmt" in err for err in result.report["errors"])
