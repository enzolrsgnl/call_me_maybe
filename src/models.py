from pydantic import BaseModel


class ParameterType(BaseModel):
    """ Model for the parameter type """
    type: str


class Function(BaseModel):
    """ Model for all function to call """
    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType


class EntryPrompt(BaseModel):
    """ Model for input prompt """
    prompt: str


class OutputResult(BaseModel):
    """ Model for the output result format """
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
