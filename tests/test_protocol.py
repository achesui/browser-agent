from pydantic import ValidationError
import pytest

from impretion_browser_agent.protocol import CreateBrowserJobRequest


def valid_request() -> dict[str, object]:
    return {
        "browser_job_id": "019c0000-0000-7000-8000-000000000001",
        "workflow_execution_id": "execution", "node_execution_id": "execution:node",
        "workflow_node_id": "node", "execution_root": "/tmp", "task": "browse",
        "headless": True, "files": [], "output_fields": [], "max_steps": 10,
        "max_actions_per_step": 2, "browser_job_token": "secret-token",
    }


def test_strict_contract_rejects_model_and_llm_configuration() -> None:
    value = valid_request()
    value["llm"] = {"model": "provider/model", "api_key": "key", "base_url": "https://bad"}
    with pytest.raises(ValidationError):
        CreateBrowserJobRequest.model_validate(value)


def test_secret_is_excluded_from_persistable_dump() -> None:
    request = CreateBrowserJobRequest.model_validate(valid_request())
    dumped = request.model_dump(mode="json", exclude={"browser_job_token"})
    assert "browser_job_token" not in dumped

