# CodeRabbit CLI Setup Guide

This guide explains how to set up and use CodeRabbit CLI for AI-powered code reviews in the Oviya EI project.

## 🚀 What is CodeRabbit?

CodeRabbit provides AI-powered code reviews that:
- Analyze code changes for bugs, performance issues, and security vulnerabilities
- Suggest improvements and best practices
- Review documentation and test coverage
- Provide contextual feedback based on your codebase

## 📋 Prerequisites

1. **GitHub Repository Access**: Admin access to configure repository secrets
2. **OpenAI API Key**: Required for CodeRabbit's AI analysis

## 🔧 Setup Instructions

### 1. Add Repository Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. CodeRabbit Configuration

The project includes two CodeRabbit configurations:

#### Main Repository (`.coderabbit.yml`)
- Reviews all code changes across the project
- Specialized for Python AI/ML and voice processing code
- Excludes documentation, tests, and build artifacts

#### Services Directory (`services/.coderabbit.yml`)
- Focused on microservices architecture
- Reviews Docker, API design, and service communication
- Optimized for production service deployment

### 3. GitHub Workflows

Two workflow files have been added:

#### `.github/workflows/coderabbit.yml`
- Triggers on pull requests to main branches
- Reviews general codebase changes
- Uses GPT-4 for comprehensive analysis

#### `services/.github/workflows/coderabbit.yml`
- Triggers on changes to services directory
- Specialized for microservices review
- Path-filtered to only review service-related changes

## 🎯 How It Works

### Automatic Reviews
- CodeRabbit automatically reviews pull requests when opened or updated
- Reviews are posted as comments on the PR
- Can request changes if critical issues are found

### Manual Reviews
- Mention `@coderabbitai review` in a comment to trigger manual review
- Useful for re-reviewing after changes

### Review Focus Areas

**OVIYA VISION VERIFICATION:**
CodeRabbit now validates that all code correctly implements Oviya's core emotional intelligence principles:

#### **Therapeutic Psychology Principles:**
- **Unconditional Positive Regard** (Carl Rogers) - Non-judgmental acceptance
- **Secure Base** (Attachment Theory) - Providing safety and exploration support
- **Vulnerability Reciprocity** - Matching user's self-disclosure level
- **Bid for Connection** - Responding to emotional connection attempts
- **Safety & Ethics** - Protecting users from harm

#### **Global Wisdom Integration:**
- **Cultural Balance** - Western psychology + global traditions (Ahimsa, Jeong, Ma, Logos, Lagom)
- **Empathic Thinking Modes** - Deep cognitive empathy across multiple modes
- **Relationship Memory** - Long-term emotional relationship tracking

#### **Voice AI Specific:**
- **Real-time Performance** - <500ms transcription, <4s end-to-end
- **GPU Memory Management** - Proper CUDA resource handling
- **Voice Data Privacy** - Encrypted, consent-based processing
- **Emotional Continuity** - Maintaining emotional context across turns

#### **Implementation Quality:**
- **Error Resilience** - Graceful failure handling in emotional processing
- **Performance Optimization** - Async patterns for real-time systems
- **Memory Safety** - Preventing leaks in long conversation sessions

## 🔍 Custom Rules

### Voice AI Specific
- **GPU Memory Management**: Flags improper CUDA memory handling
- **Async Error Handling**: Ensures WebSocket handlers have proper exception handling
- **Model Loading Validation**: Suggests validation after model loading
- **Security Headers**: Reminds about security for voice data APIs

### Services Specific
- **Docker Security**: Ensures specific image tags are used
- **Health Checks**: Validates health check configuration
- **API Error Responses**: Ensures consistent error formats
- **Inter-service Communication**: Suggests retry logic and circuit breakers

## 📊 Review Process

1. **Pull Request Created/Updated** → CodeRabbit automatically starts review
2. **Code Analysis** → AI analyzes changes for issues and improvements
3. **Comments Posted** → Review comments appear on PR
4. **Suggestions Made** → Code improvement suggestions provided
5. **Approval/Changes** → Developer addresses feedback

## ⚙️ Configuration Customization

### Modifying Review Rules

Edit `.coderabbit.yml` or `services/.coderabbit.yml`:

```yaml
custom_rules:
  - name: "Your Rule Name"
    pattern: "regex_pattern"
    message: "Your feedback message"
    level: info|warning|error
```

### Adjusting Path Filters

```yaml
path_filters:
  include:
    - "your/include/pattern/**"
  exclude:
    - "your/exclude/pattern/**"
```

### Performance Thresholds

```yaml
performance:
  max_function_lines: 50
  max_file_lines: 500
  max_cyclomatic_complexity: 10
```

## 🔒 Security Considerations

- **API Keys**: Never commit OpenAI API keys to code
- **Rate Limits**: Monitor OpenAI API usage
- **Data Privacy**: CodeRabbit only analyzes code diffs, not sensitive data
- **Permissions**: CodeRabbit only needs read access to PRs and write access for comments

## 📈 Monitoring and Maintenance

### Check Review Status
- View workflow runs in GitHub Actions tab
- Review comments appear directly on pull requests
- Check CodeRabbit's analysis in PR conversations

### Update Configuration
```bash
# Test configuration locally (if using CodeRabbit CLI locally)
coderabbit review --config .coderabbit.yml --dry-run
```

### Troubleshooting
- **No Reviews**: Check if OPENAI_API_KEY secret is set
- **Failed Workflows**: Check GitHub Actions logs
- **Incorrect Reviews**: Adjust custom rules in configuration

## 🎉 Getting Started

1. ✅ Add OPENAI_API_KEY to repository secrets
2. ✅ Configuration files are already in place
3. ✅ Workflows are ready to trigger on next PR

Create a test pull request to see CodeRabbit in action!

## 📚 Additional Resources

- [CodeRabbit Documentation](https://docs.coderabbit.ai)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [OpenAI API Best Practices](https://platform.openai.com/docs/introduction)
