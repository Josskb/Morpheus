# Deployment Guide: Financial-Radar on Proxmox

This guide walks through deploying financial-radar on a local Proxmox VM to start collecting real trading data for ML model training.

## Prerequisites

- Proxmox server with available resources (2 vCPU, 4GB RAM, 50GB disk)
- Ubuntu 22.04 Cloud Image (preferred)
- Twitter API v2 Bearer Token (free tier: 500k tweets/month)
- Optional: Telegram bot token for alerts

## Phase 1: Create Proxmox VM (15 min)

### In Proxmox Console

1. **Create VM**
   - ID: 100-110 (any available)
   - Name: `financial-radar` or similar
   - OS Type: Linux / Ubuntu 22.04
   - Disk: 50 GB (virtio-scsi)
   - vCPU: 2 cores
   - Memory: 4096 MB
   - Network: Bridged (standard bridge)

2. **First Boot Setup**
   - Boot from Ubuntu Cloud Image
   - Wait for system to configure (~2-3 min)
   - Note the IP address shown on console

### Get VM IP

```bash
# Via Proxmox console
sudo ip addr show

# Or check your router's DHCP client list
```

## Phase 2: SSH Access & Base Setup (10 min)

### SSH into VM

```bash
ssh ubuntu@192.168.1.100    # Replace with actual IP
```

### Install Docker & Docker Compose

```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
```

Logout and log back in for group changes:
```bash
exit
ssh ubuntu@192.168.1.100
```

### Verify Installation

```bash
docker --version
docker-compose --version
```

## Phase 3: Clone & Configure Application (15 min)

### Clone Repository

```bash
cd ~
git clone <YOUR_REPO_URL> financial-radar
cd financial-radar
```

### Configure Environment

```bash
# Copy production config template
cp .env.production .env

# Edit with your credentials
nano .env
```

**Required credentials to fill in:**
- `TWITTER_BEARER_TOKEN` — from developer.twitter.com
- `TELEGRAM_BOT_TOKEN` — from @BotFather (optional)
- `TELEGRAM_CHAT_ID` — your chat ID (optional)

**Optional but recommended:**
- `POSTGRES_PASSWORD` — change from default (`radar_prod_pass`)

### Update Twitter Accounts

Edit `config/accounts.yaml` to add or modify the accounts you want to monitor:

```yaml
accounts:
  - username: WClemente_
    display_name: "Will Clemente"
    markets: [crypto]
    tags: [on-chain, bitcoin]
    priority: high
    enabled: true
  # ... more accounts ...
```

Pre-configured accounts (8-10 crypto analysts):
- WClemente_ (on-chain analysis)
- CryptoCapo_ (technical analysis)
- pentosh1 (cycle analysis)
- CryptoKaleo (market structure)
- TheCryptoDog (altcoins)
- trendtraderdog (technical)
- glassnode (on-chain metrics)
- And others...

**Note:** All accounts are pre-configured with 30-minute polling interval for free tier API limits.

## Phase 4: Deploy Application (10 min)

### Start Services

```bash
cd ~/financial-radar
docker-compose up -d
```

### Verify Startup

```bash
# Check running containers
docker-compose ps

# Monitor logs
docker-compose logs -f
```

Expected output after ~30 seconds:
```
radar-postgres  | ...postgres started...
radar           | AlertService démarré (seuil=0.65, cooldown=5min, intervalle=30s).
```

Press `Ctrl+C` to exit logs (services continue running).

## Phase 5: Validation (30 min)

### Check Database Connection

```bash
docker-compose exec postgres psql -U radar -d radar -c "SELECT COUNT(*) FROM tweets;"
```

Expected output (starts at 0, increases as tweets are collected):
```
 count
-------
     0
```

### Monitor Tweet Collection

```bash
# Watch tweets being collected in real-time
watch -n 5 'docker-compose exec postgres psql -U radar -d radar -c "SELECT COUNT(*) FROM tweets;"'
```

After 5-10 minutes, should see tweets arriving:
- ~1-2 tweets per minute
- ~100-150 per hour
- ~2400-3600 per day

### Monitor NLP Processing

```bash
# Check how many tweets have been NLP processed
docker-compose exec postgres psql -U radar -d radar -c "SELECT COUNT(*) FROM tweets WHERE nlp_processed=true;"
```

Should increase every 30 seconds as the NLP processor runs.

### Monitor Market Snapshots

```bash
# Check hourly price snapshots
docker-compose exec postgres psql -U radar -d radar -c "SELECT COUNT(*) FROM market_snapshots;"
```

Increases every hour. After 1-2 hours, should have 1-2 snapshots per cryptocurrency tracked.

### Test Telegram Alerts (Optional)

1. Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
2. Reload service: `docker-compose restart radar`
3. Wait for a high-confidence alert (confidence > 0.65)
4. Check your Telegram chat for incoming alert

Alert format includes:
- Trade direction (LONG 📈 or SHORT 📉)
- Confidence level (FORT/MOYEN/FAIBLE)
- Ticker symbols
- Target price and stop loss
- Link to original tweet

## Operations

### View Logs

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines of radar app
docker-compose logs radar --tail=100

# Filter by service
docker-compose logs postgres    # Database logs
docker-compose logs radar       # App logs
```

### Database Inspection

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U radar -d radar

# Useful queries
\dt                          # List all tables
SELECT COUNT(*) FROM tweets;
SELECT COUNT(*) FROM accounts;
SELECT COUNT(*) FROM market_snapshots;

# Exit psql
\q
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart radar

# Stop services (preserves data)
docker-compose down

# Start services again
docker-compose up -d
```

### Backup Database

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U radar radar > backup_$(date +%Y%m%d).sql

# Restore from backup
cat backup_20260527.sql | docker-compose exec -T postgres psql -U radar radar
```

## Data Accumulation Timeline

With 8-10 accounts and 30-minute polling interval:

- **Week 1**: ~1000 tweets, ~300 alerts, establish baseline
- **Week 2-3**: ~3000 tweets total, ~900 alerts, patterns emerge
- **Week 4+**: Sufficient data to train XGBoost model (Module 4)

Target: **2-3 weeks minimum** of continuous collection before ML model training.

## Troubleshooting

### PostgreSQL Won't Start

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify disk space
df -h

# Reset (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### No Tweets Being Collected

1. Verify Twitter credentials in `.env`
2. Check collector logs: `docker-compose logs radar | grep -i collector`
3. Verify account list in `config/accounts.yaml`
4. Test API manually: 
   ```bash
   docker-compose exec radar python -c "
   from src.collector.service import CollectorService
   import asyncio
   svc = CollectorService()
   print(asyncio.run(svc.poll_once()))
   "
   ```

### High CPU/Memory Usage

- Check container resource limits: `docker stats`
- Reduce polling frequency if needed (edit `config/accounts.yaml`)
- Increase VM memory if persistent

### Telegram Alerts Not Sending

1. Verify `TELEGRAM_BOT_TOKEN` is set correctly
2. Verify `TELEGRAM_CHAT_ID` is correct (should be numeric)
3. Test bot connectivity: `docker-compose logs radar | grep -i telegram`
4. Check confidence threshold (default 0.65)

## Next Steps

### Week 3-4: Data Ready

Once you have 2-3 weeks of data:
1. Export dataset: `docker-compose exec postgres pg_dump -U radar radar > training_data.sql`
2. Implement Module 4 (XGBoost reliability scoring)
3. Start training ML model

### Module 4: ML Scoring (Week 4-5)

- Add XGBoost model training pipeline
- Evaluate predictions against actual market movement
- Integrate confidence scoring into alerts

### Module 6: Dashboard (Week 5-6)

- React/Vue frontend for live monitoring
- Historical performance charts
- Account performance comparison

## Support

Check logs for detailed error messages:
```bash
docker-compose logs -f
```

For issues, inspect:
- Application logs: `docker-compose logs radar --tail=200`
- Database logs: `docker-compose logs postgres`
- System resources: `docker stats`
