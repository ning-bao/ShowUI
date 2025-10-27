# Environment Configuration Guide

This guide explains how to configure the annotation pipeline using environment variables.

## Quick Setup

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your OpenAI API key:**
   ```bash
   nano .env  # or use your favorite editor
   ```

3. **Set your OpenAI API key:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

## Environment Variables

### OpenAI Configuration

#### `OPENAI_API_KEY` (Required for AI annotation)
Your OpenAI API key. Get one from https://platform.openai.com/api-keys

```bash
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

**Note:** If you don't set this or leave it as `your_openai_api_key_here`, the system will automatically use the dummy mode (random annotations for testing).

#### `OPENAI_MODEL` (Optional)
The OpenAI model to use for annotation.

- **Default:** `gpt-4o`
- **Options:** `gpt-4o`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`

```bash
OPENAI_MODEL=gpt-4o
```

**Recommendation:** Use `gpt-4o` for best results with vision tasks.

#### `OPENAI_MAX_TOKENS` (Optional)
Maximum tokens for API response.

- **Default:** `4096`
- **Range:** `1000` - `16000`

```bash
OPENAI_MAX_TOKENS=4096
```

### Flask Configuration

#### `FLASK_ENV` (Optional)
Flask environment mode.

- **Default:** `development`
- **Options:** `development`, `production`

```bash
FLASK_ENV=development
```

#### `FLASK_DEBUG` (Optional)
Enable/disable Flask debug mode.

- **Default:** `True`
- **Options:** `True`, `False`

```bash
FLASK_DEBUG=True
```

**Warning:** Set to `False` in production!

#### `SECRET_KEY` (Required for production)
Secret key for Flask session security.

```bash
SECRET_KEY=your-random-secret-key-generate-with-python-secrets
```

Generate a secure key:
```python
python -c "import secrets; print(secrets.token_hex(32))"
```

### Application Settings

#### `MAX_FILE_SIZE_MB` (Optional)
Maximum file size for uploads in megabytes.

- **Default:** `16`
- **Range:** `1` - `100`

```bash
MAX_FILE_SIZE_MB=16
```

#### `ALLOWED_EXTENSIONS` (Optional)
Comma-separated list of allowed file extensions.

- **Default:** `jpg,jpeg,png,gif,bmp`

```bash
ALLOWED_EXTENSIONS=jpg,jpeg,png,gif,bmp
```

#### `RATE_LIMIT_PER_MINUTE` (Optional)
API rate limit per minute (future feature).

- **Default:** `60`

```bash
RATE_LIMIT_PER_MINUTE=60
```

## Complete Example

Here's a complete `.env` file with all settings:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=4096

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

# Application Settings
MAX_FILE_SIZE_MB=16
ALLOWED_EXTENSIONS=jpg,jpeg,png,gif,bmp

# Optional: API Rate Limiting
RATE_LIMIT_PER_MINUTE=60
```

## Testing Configuration

### Test if OpenAI is configured correctly:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', os.getenv('OPENAI_API_KEY')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'Not set')"
```

### Test the annotator:

```python
from dotenv import load_dotenv
from utils.annotator import GPTAnnotator

load_dotenv()

# This will show if API is configured
annotator = GPTAnnotator()
print("Annotator initialized successfully!")
```

## Modes of Operation

### 1. Dummy Mode (No API Key)
```bash
# Leave API key as default or don't set it
OPENAI_API_KEY=your_openai_api_key_here
```

**Behavior:** Generates random annotations for testing. No API calls, no costs.

### 2. OpenAI API Mode (With API Key)
```bash
# Set your actual API key
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

**Behavior:** Uses OpenAI API for intelligent annotation. Costs apply per API call.

## Security Best Practices

1. **Never commit `.env` to git:**
   ```bash
   # Already in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use different keys for dev/prod:**
   - Development: Use a restricted API key with limits
   - Production: Use a separate key with monitoring

3. **Rotate keys regularly:**
   - Change your API key every 3-6 months
   - Revoke old keys after rotation

4. **Set spending limits:**
   - Configure usage limits in OpenAI dashboard
   - Set up billing alerts

5. **Restrict file upload sizes:**
   - Keep `MAX_FILE_SIZE_MB` reasonable (16-32 MB)
   - Prevents abuse and excessive API costs

## Troubleshooting

### "Running in DUMMY mode" message on startup

**Cause:** API key not set or invalid.

**Solution:**
1. Check `.env` file exists
2. Verify `OPENAI_API_KEY` is set correctly
3. Ensure no extra spaces or quotes

### "OpenAI API error" during annotation

**Possible causes:**
- Invalid API key
- Insufficient quota/credits
- Network issues
- Image too large

**Solution:**
1. Check API key is valid
2. Check OpenAI account has credits
3. Try with a smaller image
4. Check OpenAI status: https://status.openai.com/

### Environment variables not loading

**Solution:**
```bash
# Verify .env file is in the project root
ls -la .env

# Check file contents (careful not to expose key!)
cat .env | grep -v "OPENAI_API_KEY"

# Restart the Flask server
```

## Cost Estimation

Using OpenAI API incurs costs. Here's an estimate:

### GPT-4o (Vision)
- **Input:** ~$2.50 per 1M tokens (~$0.003 per image)
- **Output:** ~$10.00 per 1M tokens

**Estimated cost per image:**
- Simple UI: ~$0.01 - $0.02
- Complex UI: ~$0.03 - $0.05

**For 1000 images:**
- Estimated: $20 - $50

**Note:** Prices may vary. Check current pricing at https://openai.com/api/pricing/

## Advanced Configuration

### Using a Proxy

```bash
# Set in your environment or .env
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"
```

### Custom Timeout

Modify `utils/annotator.py` to add timeout:

```python
self.client = OpenAI(
    api_key=self.api_key,
    timeout=60.0  # seconds
)
```

### Using Azure OpenAI

Modify `utils/annotator.py`:

```python
from openai import AzureOpenAI

self.client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
```

## Support

For issues related to:
- **OpenAI API:** https://platform.openai.com/docs
- **Environment setup:** Check this guide
- **Application errors:** See main README.md



