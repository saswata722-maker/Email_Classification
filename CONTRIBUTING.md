# Contributing Guidelines

Thank you for considering contributing to the Email Spam Classifier project!

## How to Contribute

1. **Fork the Repository**: Click "Fork" at the top right of the repository page.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**: Ensure your code follows PEP 8 conventions.
5. **Test your changes**: Run the pipeline and make sure syntax checks pass:
   ```bash
   python -m py_compile Email_logistic_regression.py
   ```
6. **Commit and push**:
   ```bash
   git commit -m "feat: add your descriptive message"
   git push origin feature/your-feature-name
   ```
7. **Submit a Pull Request**: Provide a clear description of the problem and the proposed solution.
