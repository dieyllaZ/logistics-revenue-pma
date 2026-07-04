"""
Q3(c) — pytest Test Suite
Run: pytest tests/test_validate.py -v
     or from project root: pytest -v
"""
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOADS_CSV, MASTER_CLEAN_CSV
from validate_data import validate


class TestSourceData:
    def test_no_missing_critical_columns(self):
        df = pd.read_csv(LOADS_CSV)
        for col in ['load_id','customer_id','route_id','revenue','weight_lbs']:
            assert df[col].isnull().sum() == 0, f"{col} has nulls"

    def test_no_duplicate_load_ids(self):
        df = pd.read_csv(LOADS_CSV)
        assert df.duplicated(subset=['load_id']).sum() == 0

    def test_revenue_is_positive(self):
        df = pd.read_csv(LOADS_CSV)
        assert (df['revenue'] > 0).all()

    def test_load_type_categories_valid(self):
        df = pd.read_csv(LOADS_CSV)
        assert set(df['load_type'].dropna().unique()).issubset({'Dry Van','Refrigerated'})

    def test_booking_type_categories_valid(self):
        df = pd.read_csv(LOADS_CSV)
        assert set(df['booking_type'].dropna().unique()).issubset({'Spot','Dedicated','Contract'})


class TestCleanDataset:
    def test_clean_columns_present(self):
        df = pd.read_csv(MASTER_CLEAN_CSV)
        for col in ['fuel_cost_status','fuel_cost_clean','profit_margin_clean',
                    'profitability_confidence','weight_per_piece',
                    'expected_rate_revenue','customer_tenure_years',
                    'revenue_potential_bucket']:
            assert col in df.columns, f"Missing: {col}"

    def test_no_implausible_profit_margins(self):
        df = pd.read_csv(MASTER_CLEAN_CSV)
        assert (df['profit_margin_clean'] < -1.0).sum() == 0

    def test_engineered_features_no_nulls(self):
        df = pd.read_csv(MASTER_CLEAN_CSV)
        for col in ['weight_per_piece','expected_rate_revenue','customer_tenure_years']:
            assert df[col].isnull().sum() == 0, f"{col} has nulls"

    def test_revenue_potential_bucket_valid(self):
        df = pd.read_csv(MASTER_CLEAN_CSV)
        assert set(df['revenue_potential_bucket'].dropna().unique()).issubset(
            {'<200K','200K-500K','500K-1M','1M+'})

    def test_fuel_cost_status_values(self):
        df = pd.read_csv(MASTER_CLEAN_CSV)
        assert set(df['fuel_cost_status'].unique()).issubset({'Valid','Misassigned','Missing'})


class TestValidationScript:
    def test_full_validation_passes(self):
        with pytest.raises(SystemExit) as exc:
            validate()
        assert exc.value.code == 0
