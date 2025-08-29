
# UI Component Breakdown

This document outlines the proposed component structure for the new React-based UI.

## Pages

- **`DashboardPage`**: The main landing page, providing an overview of all experiment runs.
- **`WizardPage`**: A multi-step wizard for creating new experiment runs.
- **`RunDetailPage`**: A page for viewing the details of a specific run.

## Components

### DashboardPage

- **`RunList`**: A sortable and filterable table displaying all runs.
  - **`RunListItem`**: A single row in the `RunList`, showing key metrics for a run.
- **`RunCharts`**: A set of charts for visualizing and comparing run performance.

### WizardPage

- **`StageSelector`**: A component for selecting artifacts from previous runs for each stage.
- **`ParameterForm`**: A form for configuring the parameters of the new run.
- **`CommandPreview`**: A component that displays the generated command for the new run.

### RunDetailPage

- **`RunHeader`**: A header displaying the name and status of the run.
- **`RunTabs`**: A set of tabs for viewing different aspects of the run.
  - **`PerformanceTab`**: A tab for viewing the performance report of the run.
  - **`LogsTab`**: A tab for viewing the logs of the run.
  - **`ArtifactsTab`**: A tab for viewing the artifacts of the run.
