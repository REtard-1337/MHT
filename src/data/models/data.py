from sqlalchemy import Column, String
from . import Base

class DataRow(Base):
    __tablename__ = 'data'

    id = Column(String, primary_key=True)
    text = Column(String)
    label = Column(String)
