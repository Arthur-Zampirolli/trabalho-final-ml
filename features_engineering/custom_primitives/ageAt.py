import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import AgeFractional, Datetime, Double
from featuretools.primitives.base import TransformPrimitive

class AgeAt(TransformPrimitive):
    """Calculates the age in years at a specific reference date.
    
    Args:
        reference_date (pd.Timestamp): The date to calculate age at
        
    Examples:
        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("2023-01-01")
        >>> age_at = AgeAt(reference_date=reference_date)
        >>> birth_dates = pd.Series(pd.to_datetime(["2000-01-01", "1990-05-15"]))
        >>> age_at(birth_dates).tolist()
        [23.0, 32.638...]
    """

    name = "age_at"
    input_types = [ColumnSchema(logical_type=Datetime), ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Double)
    uses_calc_time = False
    description_template = "the age from {} at reference date"

    def get_function(self):
        def age_at(birth_dates, reference_dates):
            days = (reference_dates - birth_dates).dt.days
            return days / 365.25
        return age_at