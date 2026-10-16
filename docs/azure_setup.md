# Azure SQL setup

Goal: every CI build loads its fairness audit into an Azure SQL database that the
Power BI dashboard reads. You do the account and portal steps (they need your login);
the schema, ETL and CI wiring are already in the repo.

Time: about 30 minutes. Cost: $0 (Azure for Students + the Azure SQL free offer).

## 1. Azure for Students account

1. Go to <https://azure.microsoft.com/free/students> and choose **Start free**.
2. Sign in with your **school email**. Verify student status. No credit card is needed.
3. You get a student subscription with credit. The free database below does not use it.

## 2. Create the free Azure SQL database

1. In the [Azure portal](https://portal.azure.com), search **SQL databases**, then **+ Create**.
2. If a banner says **"Want to try Azure SQL Database for free?"**, click **Apply offer**.
   This is the free offer: up to 10 databases, 32 GB each, a monthly compute allowance.
3. **Basics** tab:
   - Subscription: *Azure for Students*
   - Resource group: **Create new**, name it `fairness-gates-rg`
   - Database name: `FairnessGates`
   - Server: **Create new**
     - Server name: something unique, e.g. `fairness-gates-<yourinitials>` (becomes `<name>.database.windows.net`)
     - Location: the closest US region
     - Authentication method: **Use SQL authentication**
     - Server admin login: e.g. `fgadmin`, plus a strong password. **Save it in your password manager.** Don't put it in any file in the repo.
   - Behavior when the free limit is reached: **Auto-pause the database until next month**. Don't choose the option that bills you.
4. **Networking** tab:
   - Connectivity method: **Public endpoint**
   - **Allow Azure services and resources to access this server: Yes**. GitHub-hosted Actions runners run in Azure, so this lets CI connect.
   - **Add current client IP address: Yes**, so your laptop and Power BI can connect.
5. **Review + create**, then **Create**. Deployment takes a few minutes.

> The free database **auto-pauses** when idle. The first connection after a pause takes
> about a minute; `warehouse/etl.py` retries for you. Power BI may show a timeout once.
> Refresh again.

## 3. Build the connection string

The ETL uses Microsoft's `mssql-python` driver. Fill in your server name, login and
password:

```
Server=tcp:<server-name>.database.windows.net,1433;Database=FairnessGates;Uid=<admin-login>;Pwd=<password>;Encrypt=yes;TrustServerCertificate=no;
```

(The portal's database page has **Connection strings → ODBC** with the same parts.)

## 4. Give it to GitHub Actions (a secret, never a file)

1. On GitHub, open the repo, then **Settings → Secrets and variables → Actions → New repository secret**.
2. Name: `FAIRNESS_SQL_CONNECTION_STRING`. Value: the connection string from step 3.
3. Save. From the next build on, the **Load audit into Azure SQL** step runs. It's skipped
   while the secret doesn't exist. The step creates the tables on first run (`--init-schema`).

## 5. Use it from your laptop (optional)

The secret only lives in your terminal session, never on disk:

```powershell
$env:FAIRNESS_SQL_CONNECTION_STRING = "<connection string from step 3>"
python -m ml.compare --dataset hmda
python -m warehouse.etl --dataset hmda --init-schema
```

## 6. Hardening (recommended before the demo)

The admin login can do anything. Create a **read-only login for Power BI**. In the
portal, open the database, go to **Query editor**, sign in as admin, and run this,
choosing your own password:

```sql
CREATE USER dashboard_reader WITH PASSWORD = '<choose a strong password>';
ALTER ROLE db_datareader ADD MEMBER dashboard_reader;
```

Use `dashboard_reader` in Power BI. Keep the admin login for the CI secret only.

## Local development without Azure

Everything also runs against SQL Server **LocalDB** (installed with Visual Studio / SQL
Server Express), which uses the same SQL engine:

```powershell
sqlcmd -S "(localdb)\MSSQLLocalDB" -Q "CREATE DATABASE FairnessGates"
$env:FAIRNESS_SQL_CONNECTION_STRING = "Server=(localdb)\MSSQLLocalDB;Database=FairnessGates;Trusted_Connection=yes;Encrypt=no;"
python -m warehouse.etl --dataset hmda --init-schema
```

To run the database test as well, point `FAIRNESS_TEST_SQL_CONNECTION_STRING` at a
separate test database and run `pytest warehouse/tests`.
