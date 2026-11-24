"""Delete-path tests for AWS, Azure and Google Cloud (fake clients, no network).

These check that each delete handler calls the right API with the right
identifiers, in the right order, and that cloud errors become DeleteError.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest
from azure.core.exceptions import HttpResponseError
from googleapiclient.errors import HttpError

from orphanaut.actions import azure_ai_deleter, azure_deleter, gcp_ai_deleter, gcp_deleter
from orphanaut.actions.deleter import DeleteError, delete_resource
from orphanaut.models import CloudProvider, CloudResource
from orphanaut.providers.azure_arm import ArmError
from orphanaut.providers.gcp_rest import GcpApiError
from orphanaut.providers.session import AzureSession, GcpSession, ProviderSession
from tests.fakes import FakeClient, FakeSession

SUB = "/subscriptions/sub1"
RG = f"{SUB}/resourceGroups/rg1/providers"


def _res(service, rtype, provider=CloudProvider.AWS, region="us-east-1", **extra):
    return CloudResource(
        resource_id=extra.get("id", "id-1"),
        name=extra.pop("name", "thing"),
        service=service,
        resource_type=rtype,
        region=region,
        status="",
        details="",
        provider=provider,
        extra=extra,
    )


# --------------------------------------------------------------------- dispatch


def _azure_session():
    return ProviderSession(
        CloudProvider.AZURE, AzureSession(credential=None, subscription_id="sub1"), "sub1", ""
    )


def _gcp_session():
    return ProviderSession(
        CloudProvider.GCP, GcpSession(credentials=None, project_id="proj1"), "proj1", ""
    )


def test_dispatch_routes_by_provider(monkeypatch):
    seen = []
    monkeypatch.setattr(azure_deleter, "delete_azure_resource", lambda s, r, p: seen.append("az"))
    monkeypatch.setattr(gcp_deleter, "delete_gcp_resource", lambda s, r, p: seen.append("gcp"))
    resource = _res("Compute", "Disk")
    delete_resource(_azure_session(), resource)
    delete_resource(_gcp_session(), resource)
    assert seen == ["az", "gcp"]


@pytest.mark.parametrize("session_factory", [_azure_session, _gcp_session])
def test_protected_resource_is_refused_before_any_api_call(monkeypatch, session_factory):
    monkeypatch.setattr(azure_deleter, "ComputeManagementClient", Mock(side_effect=AssertionError))
    monkeypatch.setattr(gcp_deleter.discovery, "build", Mock(side_effect=AssertionError))
    resource = _res("Compute", "Managed Disk")
    resource.deletable = False
    with pytest.raises(DeleteError, match="in use or protected"):
        delete_resource(session_factory(), resource)


# ------------------------------------------------------------------------- AWS


@pytest.mark.parametrize(
    ("service", "rtype", "extra", "client", "op", "kwargs"),
    [
        ("EC2", "Instance", {"instance_id": "i-1"}, "ec2", "terminate_instances",
         {"InstanceIds": ["i-1"]}),
        ("EBS", "Volume", {"volume_id": "vol-1"}, "ec2", "delete_volume", {"VolumeId": "vol-1"}),
        ("EBS", "Snapshot", {"snapshot_id": "snap-1"}, "ec2", "delete_snapshot",
         {"SnapshotId": "snap-1"}),
        ("EC2", "Elastic IP", {"allocation_id": "eipalloc-1"}, "ec2", "release_address",
         {"AllocationId": "eipalloc-1"}),
        ("EC2", "Elastic IP", {"public_ip": "1.2.3.4"}, "ec2", "release_address",
         {"PublicIp": "1.2.3.4"}),
        ("VPC", "NAT Gateway", {"nat_gateway_id": "nat-1"}, "ec2", "delete_nat_gateway",
         {"NatGatewayId": "nat-1"}),
        ("ELB", "Classic Load Balancer", {"load_balancer_name": "clb"}, "elb",
         "delete_load_balancer", {"LoadBalancerName": "clb"}),
        ("ELB", "Application Load Balancer", {"load_balancer_arn": "arn:alb"}, "elbv2",
         "delete_load_balancer", {"LoadBalancerArn": "arn:alb"}),
        ("RDS", "DB Instance", {"db_instance_identifier": "db1"}, "rds", "delete_db_instance",
         {"DBInstanceIdentifier": "db1", "SkipFinalSnapshot": True,
          "DeleteAutomatedBackups": True}),
        ("Lambda", "Function", {"function_name": "fn"}, "lambda", "delete_function",
         {"FunctionName": "fn"}),
        ("ECS", "Cluster", {"cluster_arn": "arn:ecs"}, "ecs", "delete_cluster",
         {"cluster": "arn:ecs"}),
        ("ElastiCache", "Cache Cluster", {"cache_cluster_id": "c1"}, "elasticache",
         "delete_cache_cluster", {"CacheClusterId": "c1"}),
        ("DynamoDB", "Table", {"table_name": "t1"}, "dynamodb", "delete_table",
         {"TableName": "t1"}),
        ("CloudWatch", "Log Group", {"log_group_name": "/lg"}, "logs", "delete_log_group",
         {"logGroupName": "/lg"}),
        ("VPC", "Interface Endpoint", {"vpc_endpoint_id": "vpce-1"}, "ec2",
         "delete_vpc_endpoints", {"VpcEndpointIds": ["vpce-1"]}),
        ("Lightsail", "Instance", {"instance_name": "ls1"}, "lightsail", "delete_instance",
         {"instanceName": "ls1"}),
    ],
)
def test_aws_simple_deletes_call_the_right_api(service, rtype, extra, client, op, kwargs):
    fake = FakeClient()
    delete_resource(FakeSession({client: fake}), _res(service, rtype, **extra))
    assert fake.calls == [(op, kwargs)]


def test_aws_global_resource_uses_us_east_1():
    session = MagicMock()
    delete_resource(session, _res("Lambda", "Function", region="global", function_name="fn"))
    session.client.assert_called_once_with("lambda", region_name="us-east-1")


def test_aws_rds_cluster_deletes_members_then_cluster():
    rds = FakeClient()
    rds.get_waiter = lambda _name: Mock()
    resource = _res("RDS", "DB Cluster", db_cluster_identifier="cl1",
                    member_instances=["m1", "m2"])
    delete_resource(FakeSession({"rds": rds}), resource)
    assert rds.ops() == ["delete_db_instance", "delete_db_instance", "delete_db_cluster"]
    assert rds.calls[-1][1]["DBClusterIdentifier"] == "cl1"


def test_aws_eks_deletes_nodegroups_and_fargate_before_cluster():
    eks = FakeClient()
    eks.get_waiter = lambda _name: Mock()
    resource = _res("EKS", "Cluster", cluster_name="k1", nodegroups=["ng1"],
                    fargate_profiles=["fp1"])
    delete_resource(FakeSession({"eks": eks}), resource)
    assert eks.ops() == ["delete_nodegroup", "delete_fargate_profile", "delete_cluster"]
    assert eks.calls[-1][1] == {"name": "k1"}


def _zone_client(records):
    return FakeClient({
        "get_hosted_zone": {"HostedZone": {"Name": "z."}},
        "list_resource_record_sets": {"ResourceRecordSets": records},
    })


def _delete_zone(route53):
    delete_resource(FakeSession({"route53": route53}),
                    _res("Route53", "Hosted Zone", region="global", hosted_zone_id="Z1"))
    return [
        change["ResourceRecordSet"]
        for op, kw in route53.calls if op == "change_resource_record_sets"
        for change in kw["ChangeBatch"]["Changes"]
    ]


def test_aws_hosted_zone_keeps_apex_ns_and_soa_until_zone_delete():
    route53 = _zone_client([
        {"Name": "z.", "Type": "NS"},
        {"Name": "z.", "Type": "SOA"},
        {"Name": "www.z.", "Type": "A"},
    ])
    deleted = _delete_zone(route53)
    assert [r["Type"] for r in deleted] == ["A"]
    assert route53.ops()[-1] == "delete_hosted_zone"


def test_aws_hosted_zone_deletes_subdomain_ns_delegation():
    route53 = _zone_client([
        {"Name": "z.", "Type": "NS"},
        {"Name": "dev.z.", "Type": "NS"},
    ])
    deleted = _delete_zone(route53)
    assert [(r["Name"], r["Type"]) for r in deleted] == [("dev.z.", "NS")]


def test_aws_hosted_zone_with_many_records_is_cleared_in_batches():
    records = [{"Name": f"r{i}.z.", "Type": "A"} for i in range(2500)]
    route53 = _zone_client(records)
    deleted = _delete_zone(route53)
    assert len(deleted) == 2500
    assert route53.ops().count("change_resource_record_sets") == 3


def test_aws_s3_bucket_is_emptied_before_delete():
    session = MagicMock()
    bucket = session.resource.return_value.Bucket.return_value
    version, obj = Mock(), Mock()
    bucket.object_versions.all.return_value = [version]
    bucket.objects.all.return_value = [obj]
    delete_resource(session, _res("S3", "Bucket", region="global", bucket_name="b1"))
    session.resource.return_value.Bucket.assert_called_once_with("b1")
    version.delete.assert_called_once()
    obj.delete.assert_called_once()
    bucket.delete.assert_called_once()


def test_aws_unknown_type_is_rejected():
    with pytest.raises(DeleteError, match="not supported"):
        delete_resource(FakeSession({}), _res("EC2", "Spaceship"))


# ----------------------------------------------------------------------- Azure


@pytest.fixture
def azure_clients(monkeypatch):
    clients = {"compute": MagicMock(), "network": MagicMock(), "storage": MagicMock()}
    monkeypatch.setattr(azure_deleter, "ComputeManagementClient",
                        lambda *_a: clients["compute"])
    monkeypatch.setattr(azure_deleter, "NetworkManagementClient",
                        lambda *_a: clients["network"])
    monkeypatch.setattr(azure_deleter, "StorageManagementClient",
                        lambda *_a: clients["storage"])
    return clients


def _az(service, rtype, **extra):
    return _res(service, rtype, CloudProvider.AZURE, "eastus", resource_group="rg1", **extra)


def test_azure_vm_delete_waits_for_completion(azure_clients):
    delete_resource(_azure_session(), _az("Compute", "Virtual Machine", vm_name="vm1"))
    vms = azure_clients["compute"].virtual_machines
    vms.begin_delete.assert_called_once_with("rg1", "vm1")
    vms.begin_delete.return_value.result.assert_called_once()


def test_azure_disk_delete(azure_clients):
    delete_resource(_azure_session(), _az("Compute", "Managed Disk", disk_name="d1"))
    azure_clients["compute"].disks.begin_delete.assert_called_once_with("rg1", "d1")


def test_azure_public_ip_delete(azure_clients):
    delete_resource(_azure_session(), _az("Network", "Public IP", ip_name="ip1"))
    azure_clients["network"].public_ip_addresses.begin_delete.assert_called_once_with(
        "rg1", "ip1"
    )


def test_azure_storage_account_delete(azure_clients):
    delete_resource(_azure_session(), _az("Storage", "Storage Account", account_name="sa1"))
    azure_clients["storage"].storage_accounts.delete.assert_called_once_with("rg1", "sa1")


def test_azure_http_error_becomes_delete_error(azure_clients):
    azure_clients["compute"].disks.begin_delete.side_effect = HttpResponseError(
        message="Disk is attached to a VM"
    )
    with pytest.raises(DeleteError, match="attached"):
        delete_resource(_azure_session(), _az("Compute", "Managed Disk", disk_name="d1"))


def test_azure_unknown_type_is_rejected(azure_clients):
    with pytest.raises(DeleteError, match="not supported"):
        delete_resource(_azure_session(), _az("Compute", "Spaceship"))


# -------------------------------------------------------------------- Azure AI


class RecordingArm:
    """ArmClient stand-in that records every delete and can fail on chosen paths."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.deletes: list[tuple[str, str, dict]] = []
        self.fail = fail or set()

    def delete(self, path, api_version, **params):
        self.deletes.append((path, api_version, params))
        if path in self.fail:
            raise ArmError(409, f"cannot delete {path}")


@pytest.fixture
def arm(monkeypatch):
    holder = {"arm": RecordingArm()}
    monkeypatch.setattr(azure_ai_deleter, "ArmClient", lambda _cred: holder["arm"])
    return holder


def test_azure_openai_deletes_deployments_then_account_then_purges(arm):
    acct = f"{RG}/Microsoft.CognitiveServices/accounts/oai"
    resource = _az("Azure OpenAI", "Account", name="oai", arm_id=acct,
                   deployments=["gpt-4o", "mini"])
    delete_resource(_azure_session(), resource)
    paths = [p for p, _, _ in arm["arm"].deletes]
    assert paths == [
        f"{acct}/deployments/gpt-4o",
        f"{acct}/deployments/mini",
        acct,
        f"{SUB}/providers/Microsoft.CognitiveServices/locations/eastus"
        "/resourceGroups/rg1/deletedAccounts/oai",
    ]


def test_azure_openai_purge_failure_is_ignored(arm):
    acct = f"{RG}/Microsoft.CognitiveServices/accounts/oai"
    purge = (f"{SUB}/providers/Microsoft.CognitiveServices/locations/eastus"
             "/resourceGroups/rg1/deletedAccounts/oai")
    arm["arm"] = RecordingArm(fail={purge})
    delete_resource(_azure_session(), _az("AI Services", "Account", name="oai", arm_id=acct))
    assert arm["arm"].deletes[-1][0] == purge


def test_azure_ml_workspace_cascade_order_and_linked_resources(arm):
    ws = f"{RG}/Microsoft.MachineLearningServices/workspaces/ws1"
    kv = f"{RG}/Microsoft.KeyVault/vaults/kv1"
    unknown = f"{RG}/Microsoft.Weird/things/w1"
    resource = _az(
        "Machine Learning", "Workspace", name="ws1", arm_id=ws,
        project_ids=[f"{RG}/Microsoft.MachineLearningServices/workspaces/proj"],
        endpoint_ids=[f"{ws}/onlineEndpoints/ep"],
        serverless_ids=[f"{ws}/serverlessEndpoints/sl"],
        compute_ids=[f"{ws}/computes/ci"],
        exclusive_ids=[kv, unknown],
    )
    delete_resource(_azure_session(), resource)
    deletes = arm["arm"].deletes
    names = [p.rsplit("/", 1)[-1] for p, _, _ in deletes]
    assert names == ["proj", "ep", "sl", "ci", "ws1", "kv1"]  # unknown type is skipped
    assert deletes[0][2] == {"forceToPurge": "true"}
    assert deletes[3][2] == {"underlyingResourceAction": "Delete"}
    assert deletes[4][2] == {"forceToPurge": "true"}
    assert deletes[5][1] == "2023-07-01"


def test_azure_ml_linked_resource_failure_is_reported_not_raised(arm):
    ws = f"{RG}/Microsoft.MachineLearningServices/workspaces/ws1"
    kv = f"{RG}/Microsoft.KeyVault/vaults/kv1"
    arm["arm"] = RecordingArm(fail={kv})
    messages = []
    delete_resource(
        _azure_session(),
        _az("Machine Learning", "Workspace", name="ws1", arm_id=ws, exclusive_ids=[kv]),
        on_progress=messages.append,
    )
    assert any("Could not delete kv1" in m for m in messages)


def test_azure_ml_standalone_compute_deletes_underlying_vm(arm):
    compute = f"{RG}/Microsoft.MachineLearningServices/workspaces/ws1/computes/ci"
    delete_resource(_azure_session(),
                    _az("Machine Learning", "Compute Instance", arm_id=compute))
    assert arm["arm"].deletes == [
        (compute, azure_ai_deleter.ML_API, {"underlyingResourceAction": "Delete"})
    ]


def test_azure_ai_arm_error_becomes_delete_error(arm):
    search = f"{RG}/Microsoft.Search/searchServices/s1"
    arm["arm"] = RecordingArm(fail={search})
    with pytest.raises(DeleteError, match="cannot delete"):
        delete_resource(_azure_session(), _az("AI Search", "Search Service", arm_id=search))


# ---------------------------------------------------------------- Google Cloud


@pytest.fixture
def gcp_build(monkeypatch):
    services = {"compute": MagicMock(), "storage": MagicMock()}
    monkeypatch.setattr(gcp_deleter.discovery, "build",
                        lambda name, *_a, **_k: services[name])
    return services


def _gc(service, rtype, **extra):
    return _res(service, rtype, CloudProvider.GCP, "us-central1", **extra)


def test_gcp_instance_delete(gcp_build):
    delete_resource(_gcp_session(),
                    _gc("Compute", "Instance", zone="us-central1-a", instance_name="vm1"))
    gcp_build["compute"].instances.return_value.delete.assert_called_once_with(
        project="proj1", zone="us-central1-a", instance="vm1"
    )


def test_gcp_disk_delete(gcp_build):
    delete_resource(_gcp_session(), _gc("Compute", "Disk", zone="us-central1-a", disk_name="d1"))
    gcp_build["compute"].disks.return_value.delete.assert_called_once_with(
        project="proj1", zone="us-central1-a", disk="d1"
    )


def test_gcp_static_ip_uses_extra_region(gcp_build):
    resource = _gc("Compute", "Static IP", address_name="ip1")
    resource.extra["region"] = "europe-west1"
    delete_resource(_gcp_session(), resource)
    gcp_build["compute"].addresses.return_value.delete.assert_called_once_with(
        project="proj1", region="europe-west1", address="ip1"
    )


def test_gcp_bucket_is_emptied_across_pages_then_deleted(gcp_build):
    objects = gcp_build["storage"].objects.return_value
    page1, page2 = Mock(), Mock()
    page1.execute.return_value = {"items": [{"name": "a"}, {"name": "b"}]}
    page2.execute.return_value = {"items": [{"name": "c"}]}
    objects.list.return_value = page1
    objects.list_next.side_effect = [page2, None]

    delete_resource(_gcp_session(), _gc("Storage", "Bucket", bucket_name="b1"))

    deleted = [c.kwargs["object"] for c in objects.delete.call_args_list]
    assert deleted == ["a", "b", "c"]
    gcp_build["storage"].buckets.return_value.delete.assert_called_once_with(bucket="b1")


def test_gcp_versioned_bucket_deletes_every_generation(gcp_build):
    objects = gcp_build["storage"].objects.return_value
    page = Mock()
    page.execute.return_value = {"items": [
        {"name": "a", "generation": "1"},
        {"name": "a", "generation": "2"},
    ]}
    objects.list.return_value = page
    objects.list_next.return_value = None

    delete_resource(_gcp_session(), _gc("Storage", "Bucket", bucket_name="b1"))

    objects.list.assert_called_once_with(bucket="b1", versions=True)
    assert [c.kwargs for c in objects.delete.call_args_list] == [
        {"bucket": "b1", "object": "a", "generation": "1"},
        {"bucket": "b1", "object": "a", "generation": "2"},
    ]


def test_gcp_http_error_becomes_delete_error(gcp_build):
    err = HttpError(Mock(status=400, reason="Bad Request"), b"disk is in use")
    gcp_build["compute"].disks.return_value.delete.return_value.execute.side_effect = err
    with pytest.raises(DeleteError):
        delete_resource(_gcp_session(),
                        _gc("Compute", "Disk", zone="us-central1-a", disk_name="d1"))


def test_gcp_unknown_type_is_rejected(gcp_build):
    with pytest.raises(DeleteError, match="not supported"):
        delete_resource(_gcp_session(), _gc("Compute", "Spaceship"))


# ------------------------------------------------------------- Google Cloud AI


class RecordingRest:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.fail = fail

    def post(self, url, body=None):
        self.calls.append(("post", url, body))
        return {"name": "op"}

    def delete(self, url, **params):
        self.calls.append(("delete", url, params))
        if self.fail:
            raise GcpApiError(403, "permission denied")
        return {"name": "op"}

    def wait(self, op, base):
        self.calls.append(("wait", base))
        return {}


@pytest.fixture
def rest(monkeypatch):
    holder = {"rest": RecordingRest()}
    monkeypatch.setattr(gcp_ai_deleter, "GcpRest", lambda *_a: holder["rest"])
    return holder


VERTEX = "https://us-central1-aiplatform.googleapis.com/v1"
EP = "projects/proj1/locations/us-central1/endpoints/123"


def test_vertex_endpoint_undeploys_models_before_delete(rest):
    delete_resource(_gcp_session(),
                    _gc("Vertex AI", "Endpoint", rest_name=EP, deployed_ids=["m1", "m2"]))
    calls = rest["rest"].calls
    assert [c[0] for c in calls] == ["post", "wait", "post", "wait", "delete", "wait"]
    assert calls[0][1] == f"{VERTEX}/{EP}:undeployModel"
    assert calls[0][2] == {"deployedModelId": "m1"}
    assert calls[4][1] == f"{VERTEX}/{EP}"


def test_vertex_index_endpoint_undeploys_indexes(rest):
    name = "projects/proj1/locations/us-central1/indexEndpoints/9"
    delete_resource(_gcp_session(),
                    _gc("Vertex AI", "Vector Search Endpoint", rest_name=name,
                        deployed_ids=["idx"]))
    assert rest["rest"].calls[0][1:] == (f"{VERTEX}/{name}:undeployIndex",
                                         {"deployedIndexId": "idx"})


def test_vertex_running_job_is_cancelled_not_deleted(rest):
    name = "projects/proj1/locations/us-central1/customJobs/7"
    delete_resource(_gcp_session(), _gc("Vertex AI", "Custom Job", rest_name=name))
    assert rest["rest"].calls == [("post", f"{VERTEX}/{name}:cancel", None)]


def test_vertex_agent_engine_is_force_deleted(rest):
    name = "projects/proj1/locations/us-central1/reasoningEngines/5"
    delete_resource(_gcp_session(), _gc("Vertex AI", "Agent Engine", rest_name=name))
    assert rest["rest"].calls[0] == ("delete", f"{VERTEX}/{name}", {"force": "true"})


def test_workbench_and_search_use_their_own_hosts(rest):
    nb = "projects/proj1/locations/us-central1-a/instances/nb1"
    ds = "projects/proj1/locations/global/collections/default_collection/dataStores/d1"
    delete_resource(_gcp_session(), _gc("Vertex AI", "Workbench Instance", rest_name=nb))
    search = _gc("Vertex AI", "Data Store", rest_name=ds)
    search.region = "global"
    delete_resource(_gcp_session(), search)
    urls = [c[1] for c in rest["rest"].calls if c[0] == "delete"]
    assert urls == [
        f"https://notebooks.googleapis.com/v2/{nb}",
        f"https://discoveryengine.googleapis.com/v1/{ds}",
    ]


def test_vertex_api_error_becomes_delete_error(rest):
    rest["rest"] = RecordingRest(fail=True)
    with pytest.raises(DeleteError, match="permission denied"):
        delete_resource(_gcp_session(), _gc("Vertex AI", "Model", rest_name=EP))
