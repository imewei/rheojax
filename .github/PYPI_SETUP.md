# PyPI Publishing Setup

This document explains how to set up PyPI trusted publishing for the RheoJAX package.

## What is Trusted Publishing?

Trusted Publishing (also called OIDC publishing) is PyPI's recommended method for publishing packages. It uses OpenID Connect (OIDC) to establish trust between GitHub Actions and PyPI, eliminating the need to store long-lived API tokens as GitHub secrets.

**Benefits:**
- More secure (no API tokens to manage or leak)
- Easier to set up and maintain
- Automatic token rotation
- Fine-grained permissions

## Setup Instructions

### 1. Configure PyPI Trusted Publishing

1. **Log in to PyPI**: Visit https://pypi.org and log in with your account

2. **Navigate to Publishing Settings**:
   - Go to your account settings
   - Select "Publishing" from the left sidebar
   - Or visit: https://pypi.org/manage/account/publishing/

3. **Add a new pending publisher**:
   - Click "Add a new pending publisher"
   - Fill in the form:
     - **PyPI Project Name**: `rheojax`
     - **Owner**: `imewei` (GitHub username/org)
     - **Repository name**: `rheojax`
     - **Workflow name**: `release.yml`
     - **Environment name**: `pypi`
   - Click "Add"

4. **Verify the pending publisher**:
   - The publisher will show as "pending" until the first successful publish
   - After the first release, it becomes an active publisher

### 2. Create GitHub Environment (Already Configured)

The workflow uses a GitHub environment named `pypi` which provides:
- Protection rules (optional: require reviewers before deployment)
- Deployment visibility
- Environment-specific secrets (if needed in future)

To configure environment protection (optional):
1. Go to repository Settings → Environments
2. Click on `pypi` environment
3. Add protection rules:
   - Required reviewers (recommended for production)
   - Wait timer
   - Deployment branches (e.g., only `main` or tags)

### 3. How to Publish a Release

Once trusted publishing is configured, publishing is automatic:

1. **Update version in `pyproject.toml`**:
   ```toml
   version = "0.2.1"
   ```

2. **Commit and push the version change**:
   ```bash
   git add pyproject.toml
   git commit -m "chore: bump version to 0.2.1"
   git push origin main
   ```

3. **Create a GitHub Release**:
   ```bash
   # Create and push tag
   git tag v0.2.1
   git push origin v0.2.1

   # Or create release via GitHub UI
   # - Go to Releases → Draft a new release
   # - Choose tag: v0.2.1 (or create new tag)
   # - Release title: v0.2.1 or "RheoJAX v0.2.1"
   # - Add release notes
   # - Click "Publish release"
   ```

4. **Automatic workflow execution**:
   - When the release is published, `release.yml` workflow triggers automatically
   - Runs full test suite on Python 3.12 and 3.13 across Ubuntu, macOS, and Windows
   - Runs linting (Black, Ruff, MyPy)
   - Builds documentation
   - Builds package distributions (sdist and wheel)
   - Verifies version matches the git tag
   - Publishes to PyPI (only if all tests pass)

### 4. Monitoring Release Progress

Track the release workflow:
1. Go to repository → Actions tab
2. Find the "Release to PyPI" workflow run
3. Monitor each job:
   - `test` (matrix: 3 OS × 2 Python versions = 6 jobs)
   - `lint` (code quality checks)
   - `docs` (documentation build)
   - `publish` (PyPI upload - only runs if all above pass)

### 5. Verification After Publishing

After successful publication:
1. Check PyPI: https://pypi.org/project/rheojax/
2. Verify the new version is listed
3. Test installation:
   ```bash
   pip install --upgrade rheojax
   python -c "import rheojax; print(rheojax.__version__)"
   ```

## Troubleshooting

### "Version mismatch" Error

If the workflow fails with version mismatch:
- **Cause**: Git tag doesn't match version in `pyproject.toml`
- **Fix**: Ensure `pyproject.toml` version matches the tag (without 'v' prefix)
  - Tag: `v0.2.1` → `pyproject.toml`: `version = "0.2.1"`

### "Trusted publisher not configured" Error

If PyPI rejects the upload:
- **Cause**: Trusted publishing not set up on PyPI
- **Fix**: Follow step 1 above to configure pending publisher on PyPI

### Tests Fail During Release

If tests fail:
- **Behavior**: `publish` job is skipped (won't upload to PyPI)
- **Fix**: Fix the failing tests and create a new patch release

### Multiple Wheels Built

If you see multiple wheels:
- **Expected**: One `.whl` (wheel) and one `.tar.gz` (source distribution)
- **Issue**: If seeing platform-specific wheels, check `pyproject.toml` build config

## Security Best Practices

1. **Never commit API tokens**: Trusted publishing eliminates this need
2. **Use environment protection**: Require manual approval for PyPI deployments
3. **Limit deployment branches**: Only allow releases from `main` or tags
4. **Monitor deployments**: Check PyPI release history regularly
5. **Enable 2FA on PyPI**: Protect your PyPI account with two-factor authentication

## Rollback Procedure

If a bad release is published:

1. **Yank the release on PyPI** (doesn't delete, marks as unavailable):
   ```bash
   # Install twine
   pip install twine

   # Yank the version
   twine yank rheojax==0.2.1 -m "Reason for yanking"
   ```

2. **Create a new fixed release**:
   - Fix the issue in code
   - Bump to next patch version (e.g., 0.2.2)
   - Create new release

**Note**: PyPI doesn't allow re-uploading the same version number. You must use a new version.

## Alternative: Manual Publishing (Not Recommended)

If you need to publish manually without GitHub Actions:

1. **Generate PyPI API token**:
   - Go to PyPI → Account Settings → API tokens
   - Create token with scope limited to `rheojax` project

2. **Configure `.pypirc`**:
   ```ini
   [pypi]
   username = __token__
   password = pypi-AgEIcH...your-token-here
   ```

3. **Build and upload**:
   ```bash
   python -m build
   twine upload dist/*
   ```

**Warning**: Manual publishing is less secure and bypasses automated testing.

## References

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions: Publishing to PyPI](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [PyPA gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)
