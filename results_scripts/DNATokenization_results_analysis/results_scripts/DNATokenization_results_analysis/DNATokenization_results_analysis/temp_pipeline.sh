# Step 7: Create results LaTeX tables
echo "Step 7: Creating results LaTeX tables..."
cd results_scripts
python create_results_latex_tables.py > create_results_latex_tables.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Created results LaTeX tables"
else
    echo "✗ Error creating results tables. Check create_results_latex_tables.log"
    exit 1
fi
cd ..

echo "========================================"
echo "Pipeline completed successfully!"
echo "Results can be found in:"
echo "  - final_combined_attn_results/"
echo "  - final_combined_ss_results/"
echo "  - ss_tables/"
echo "  - ss_figures/"
echo "  - results_tables/"
