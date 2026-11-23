# AWS deployment (ECS Fargate)

> **Status: written but not yet run against AWS (untested).** Update this note and replace the
> "Expected" lines with real output once each step has been run.

This covers spec §3.9: deploying the Flask API container to AWS with ECR, ECS Fargate, Secrets Manager, CloudWatch and IAM, deployed by GitHub Actions.

## Architecture

```
GitHub (main) --CI passes--> Deploy workflow --(manual approval: "production" environment)-->
   OIDC -> IAM role telco-churn-github-deploy (least privilege, no stored keys)
   -> docker build -> ECR telco-churn-api:<git sha>
   -> ECS task definition (Fargate 0.5 vCPU / 1 GB) -> ECS service telco-churn-api (1 task, public IP)
Task at startup: Secrets Manager (DagsHub token) -> MLflow registry on DagsHub -> loads model alias `production`
Logs: awslogs -> CloudWatch log group /ecs/telco-churn-api (7-day retention) -> metric filter + alarms
Access: security group allows TCP 5000 only from the team's IP (/32)
```

| Decision | Why |
|---|---|
| Region `ap-southeast-2` only | Account constraint; keeps everything in one place |
| Fargate 512 CPU / 1024 MB | Measured API memory ~430–460 MiB in Docker (`docker stats`); 0.25 vCPU / 0.5 GB would be too tight |
| Public subnet + public IP, no load balancer, no NAT gateway | Cheapest option that can pull from ECR and reach DagsHub |
| Model loaded at startup from the MLflow registry (`MODEL_ALIAS=production`) | Same image works for any approved model; promotion and rollback move the alias and restart the task |
| DagsHub token in Secrets Manager, injected as env vars | Never in the image, the repo or GitHub |
| GitHub OIDC instead of AWS access keys | No long-lived AWS keys anywhere; the role trusts only this repo's `production` environment |
| Deployment circuit breaker with rollback | If a new task never becomes healthy, ECS returns to the previous task definition automatically |
| Health check uses `/health` with a 180 s start period | `/health` returns 503 until the model is downloaded (20–60 s observed locally) |

Files: `deploy/aws/task-definition.json`, `deploy/aws/iam/*.json`, `.github/workflows/deploy.yml`.

---

## One-time setup

Run steps 1–8 in **AWS CloudShell**. Open the AWS console, check that the region (top right) is **Asia Pacific (Sydney) ap-southeast-2**, then click the CloudShell icon `>_`. CloudShell already has the AWS CLI, git and jq, and uses your console login, so **no access keys are created**.

Do this as an IAM user with admin rights, not the root user, if one exists. If you only have root, enable MFA on it first.

### 1. Get the code and confirm the account
```bash
git clone https://github.com/Muhammad-Ahmad2511/telco-churn-mlops.git
cd telco-churn-mlops
export AWS_REGION=ap-southeast-2
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws sts get-caller-identity
aws budgets describe-budgets --account-id "$ACCOUNT_ID" \
  --query 'Budgets[].{name:BudgetName,limit:BudgetLimit.Amount,unit:BudgetLimit.Unit}' --output table
```
Expected: your account and user ARN, then at least one budget. **If no budget is listed, create one first** (Billing → Budgets). §3.9 requires a budget alert before deployment.

### 2. Container registry (ECR)
```bash
aws ecr create-repository --repository-name telco-churn-api \
  --image-scanning-configuration scanOnPush=true --query repository.repositoryUri --output text
aws ecr put-lifecycle-policy --repository-name telco-churn-api --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep last 5 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}'
```
`scanOnPush` gives a second vulnerability scan in addition to Trivy in CI. The lifecycle policy limits storage cost.

### 3. CloudWatch log group (7-day retention)
```bash
aws logs create-log-group --log-group-name /ecs/telco-churn-api
aws logs put-retention-policy --log-group-name /ecs/telco-churn-api --retention-in-days 7
```

### 4. DagsHub token in Secrets Manager
The token is typed in hidden and never appears on screen or in shell history:
```bash
read -rp  "DagsHub username: " DH_USER
read -rsp "DagsHub token: "    DH_TOKEN; echo
aws secretsmanager create-secret --name telco-churn/dagshub \
  --secret-string "$(jq -n --arg u "$DH_USER" --arg t "$DH_TOKEN" '{username:$u,token:$t}')" \
  --query ARN --output text
unset DH_TOKEN
```
**Copy the printed ARN.** It becomes the GitHub secret `DAGSHUB_SECRET_ARN` in step 7.

### 5. ECS task execution role (pulls the image, writes logs, reads only that one secret)
```bash
aws iam create-role --role-name telco-churn-ecs-execution \
  --assume-role-policy-document file://deploy/aws/iam/ecs-tasks-trust-policy.json
aws iam attach-role-policy --role-name telco-churn-ecs-execution \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" deploy/aws/iam/ecs-execution-secret-policy.json > /tmp/exec-secret.json
aws iam put-role-policy --role-name telco-churn-ecs-execution \
  --policy-name read-dagshub-secret --policy-document file:///tmp/exec-secret.json
```

### 6. GitHub OIDC provider and the deploy role
```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com
sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" deploy/aws/iam/github-oidc-trust-policy.json > /tmp/gh-trust.json
aws iam create-role --role-name telco-churn-github-deploy \
  --assume-role-policy-document file:///tmp/gh-trust.json --query Role.Arn --output text
sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" deploy/aws/iam/github-deploy-policy.json > /tmp/gh-deploy.json
aws iam put-role-policy --role-name telco-churn-github-deploy \
  --policy-name deploy-telco-churn --policy-document file:///tmp/gh-deploy.json
```
**Copy the printed role ARN.** It becomes the GitHub secret `AWS_DEPLOY_ROLE_ARN`.

If `create-open-id-connect-provider` says the provider already exists, skip that command. If it asks for `--thumbprint-list`, check the current AWS documentation for "Create an OpenID Connect identity provider" for GitHub Actions; this step is not verified.

The trust policy only lets **this repository's `production` environment** assume the role, and the permissions only allow pushing to one ECR repository and updating one ECS service.

### 7. ECS cluster, network and security group
First, on your **laptop** (PowerShell), get your public IP:
```powershell
(Invoke-RestMethod https://checkip.amazonaws.com).Trim()
```
Then in CloudShell, replacing `x.x.x.x` with that IP:
```bash
MY_IP=x.x.x.x
aws ecs create-cluster --cluster-name telco-churn
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" Name=default-for-az,Values=true \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
SG=$(aws ec2 create-security-group --group-name telco-churn-api \
  --description "Churn API: port 5000 from the team IP only" --vpc-id "$VPC" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 5000 --cidr "$MY_IP/32"
echo "ECS_SUBNETS=$SUBNETS"
echo "ECS_SECURITY_GROUP=$SG"
```
If `VPC` prints `None`, the account has no default VPC: create one with `aws ec2 create-default-vpc` and repeat.
If your home IP changes later, add the new IP with the same `authorize-security-group-ingress` command.

### 8. GitHub settings (browser)
1. **Settings → Environments → New environment** → name `production`.
   - **Required reviewers:** add yourself (and teammates). This is the manual approval gate before every deployment.
   - **Environment secrets:** `AWS_DEPLOY_ROLE_ARN` (step 6) and `DAGSHUB_SECRET_ARN` (step 4).
2. **Settings → Secrets and variables → Actions → Variables tab** → add `ECS_SUBNETS` and `ECS_SECURITY_GROUP` (step 7), and `DEPLOY_ENABLED` = `true`. Until `DEPLOY_ENABLED` is set, the Deploy workflow is skipped.

---

## Deploying

- **Automatic:** every merge to `main` runs CI; if CI passes, the Deploy workflow starts and **waits for approval**.
- **Manual:** Actions → Deploy → Run workflow → `main`.

Approve it (Review deployments → production → Approve). The job pushes `telco-churn-api:<git sha>` to ECR, registers a new task definition, creates or updates the service, and waits until the task is **HEALTHY**. Health only turns HEALTHY once `/health` returns 200, which means the production model has loaded from the registry.

### 9. Find the endpoint and test it
In CloudShell:
```bash
TASK=$(aws ecs list-tasks --cluster telco-churn --service-name telco-churn-api --query 'taskArns[0]' --output text)
ENI=$(aws ecs describe-tasks --cluster telco-churn --tasks "$TASK" \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)
aws ec2 describe-network-interfaces --network-interface-ids "$ENI" \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text
```
On your laptop (PowerShell), with that IP:
```powershell
$API = "http://<public-ip>:5000"
Invoke-RestMethod "$API/health"
Invoke-RestMethod "$API/model-info"
$body = @{SeniorCitizen=0; Partner="Yes"; Dependents="No"; tenure=12; PhoneService="Yes"; MultipleLines="No"; InternetService="DSL"; OnlineSecurity="Yes"; OnlineBackup="No"; DeviceProtection="No"; TechSupport="No"; StreamingTV="No"; StreamingMovies="No"; Contract="Month-to-month"; PaperlessBilling="Yes"; PaymentMethod="Electronic check"; MonthlyCharges=29.85; TotalCharges=358.2} | ConvertTo-Json
Invoke-RestMethod "$API/predict" -Method Post -ContentType "application/json" -Body $body
.venv\Scripts\python -m src.monitoring.traffic --url $API --n 30
```
Access-control check: the same URL from a phone on mobile data (a different IP) should **time out**.

The public IP changes whenever the task is replaced. Do not commit it, because the repository is public.

### 10. CloudWatch logs and alarms
```bash
aws logs tail /ecs/telco-churn-api --since 15m
aws cloudwatch put-metric-alarm --alarm-name telco-churn-api-high-cpu \
  --namespace AWS/ECS --metric-name CPUUtilization \
  --dimensions Name=ClusterName,Value=telco-churn Name=ServiceName,Value=telco-churn-api \
  --statistic Average --period 60 --evaluation-periods 3 --threshold 80 \
  --comparison-operator GreaterThanThreshold --treat-missing-data notBreaching
aws logs put-metric-filter --log-group-name /ecs/telco-churn-api --filter-name api-error-logs \
  --filter-pattern '"ERROR"' \
  --metric-transformations metricName=ApiErrorLogs,metricNamespace=TelcoChurn,metricValue=1,defaultValue=0
aws cloudwatch put-metric-alarm --alarm-name telco-churn-api-error-logs \
  --namespace TelcoChurn --metric-name ApiErrorLogs --statistic Sum --period 300 \
  --evaluation-periods 1 --threshold 5 --comparison-operator GreaterThanOrEqualToThreshold \
  --treat-missing-data notBreaching
```
Logs contain gunicorn access lines (path, status, timing) and `predict instances=… model_version=… latency_ms=…`. They never contain request bodies.

---

## Rollback

| What went wrong | Rollback |
|---|---|
| Bad **model** | On the laptop: `python -m src.models.promote rollback`, then in CloudShell: `aws ecs update-service --cluster telco-churn --service telco-churn-api --force-new-deployment`. The new task loads the previous `production` version |
| Bad **code/image** | `aws ecs update-service --cluster telco-churn --service telco-churn-api --task-definition telco-churn-api:<previous revision>`, or re-run Deploy on the previous commit |
| New task never healthy | Automatic: the deployment circuit breaker rolls back to the last working task definition |

---

## Cost control and cleanup

- Check prices on the official AWS pricing pages; no prices are assumed here. Billable items: the Fargate task while running, its public IPv4 address, the Secrets Manager secret, ECR storage and CloudWatch logs.
- **Pause** (no running task) after each demo:
  ```bash
  aws ecs update-service --cluster telco-churn --service telco-churn-api --desired-count 0
  ```
- **Delete everything** after the final demo (§3.9):
  ```bash
  aws ecs update-service --cluster telco-churn --service telco-churn-api --desired-count 0
  aws ecs delete-service --cluster telco-churn --service telco-churn-api --force
  aws ecs delete-cluster --cluster telco-churn
  aws ecr delete-repository --repository-name telco-churn-api --force
  aws logs delete-log-group --log-group-name /ecs/telco-churn-api
  aws cloudwatch delete-alarms --alarm-names telco-churn-api-high-cpu telco-churn-api-error-logs
  aws secretsmanager delete-secret --secret-id telco-churn/dagshub --force-delete-without-recovery
  aws iam delete-role-policy --role-name telco-churn-github-deploy --policy-name deploy-telco-churn
  aws iam delete-role --role-name telco-churn-github-deploy
  aws iam delete-role-policy --role-name telco-churn-ecs-execution --policy-name read-dagshub-secret
  aws iam detach-role-policy --role-name telco-churn-ecs-execution \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  aws iam delete-role --role-name telco-churn-ecs-execution
  aws iam delete-open-id-connect-provider --open-id-connect-provider-arn \
    "arn:aws:iam::$ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
  aws ec2 delete-security-group --group-id "$SG"
  ```
  Task definition revisions are free and can stay registered. Take a screenshot of the empty ECS and ECR consoles as cleanup evidence.

## Environment variables and secrets (names only)

| Name | Where | Purpose |
|---|---|---|
| `AWS_DEPLOY_ROLE_ARN` | GitHub environment secret (`production`) | IAM role assumed through OIDC |
| `DAGSHUB_SECRET_ARN` | GitHub environment secret (`production`) | Secrets Manager ARN injected into the task definition |
| `ECS_SUBNETS`, `ECS_SECURITY_GROUP` | GitHub repository variables | Network for the first `create-service` |
| `DEPLOY_ENABLED` | GitHub repository variable | `true` turns on automatic deployment |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Secrets Manager → container env | DagsHub MLflow registry access |
| `MODEL_ALIAS`, `MLFLOW_TRACKING_URI`, `LOG_LEVEL` | Task definition (not secret) | Which model to serve and where to load it from |
