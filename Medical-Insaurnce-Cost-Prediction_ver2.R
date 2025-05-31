install.packages("caret")
install.packages("MASS")
install.packages(c("ggplot2", "dplyr", "reshape2", "GGally"))
install.packages("corrplot")
install.packages("glmnet")
install.packages("lmtest")
install.packages("randomForest")
install.packages("shiny")



library(ggplot2)
library(GGally)
library(caret)
library(MASS)
library(glmnet)
library(corrplot)
library(dplyr)
library(lmtest)
library(randomForest)
library(shiny)



# Load the data
insurance_data <- read.csv("insurance.csv")
str(insurance_data)
summary(insurance_data)


# Plots
ggpairs(insurance_data, aes(color = insurance_data$region))
# Scatterplot with regression line
ggplot(insurance_data, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal() +
  scale_color_manual(values = c("no" = "blue", "yes" = "red")) +
  labs(title = "Age vs. Charges with Smoker Status", x = "Age", y = "Charges")
# Boxplot: Smoking vs. Charges
ggplot(insurance_data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot() +
  labs(title = "Smoker vs. Insurance Charges", x = "Smoker", y = "Charges")
# Correlation heatmap
numeric_data <- insurance_data[, sapply(insurance_data, is.numeric)]
corr_matrix <- cor(numeric_data)
corrplot(corr_matrix, method = "color", addCoef.col = "black")
# Densityplot
ggplot(insurance_data, aes(x = charges)) +
  geom_density(fill = "green", alpha = 0.5) +
  labs(title = "Density Plot of Charges", x = "Charges")
# Pair plot
ggpairs(insurance_data, columns = c("age", "bmi", "charges"),
        aes(color = smoker, alpha = 0.7)) +
  theme_minimal() +
  ggtitle("Pair Plot of Age, BMI, and Charges")
# Scatter Plots
plot(predict(model), residuals(model), main = "Residuals vs Predicted")
abline(h = 0, col = "red")
# Normality of Residuals
qqnorm(residuals(model))
qqline(residuals(model), col = "blue")
shapiro.test(residuals(model))


# Convert categorical variables into factors
insurance_data$smoker <- as.factor(insurance_data$smoker)
insurance_data$region <- as.factor(insurance_data$region)
# Check for missing values
colSums(is.na(insurance_data))
# Labeling the BMI Catogories
insurance_data <- insurance_data %>%
  mutate(bmi_category = cut(bmi, breaks = c(0, 18.5, 24.9, 29.9, Inf),
                            labels = c("Underweight", "Normal", "Overweight", "Obese")))
ggplot(insurance_data, aes(x = bmi_category, y = charges, fill = smoker)) +
  geom_boxplot() +
  scale_fill_manual(values = c("no" = "blue", "yes" = "red")) +
  theme_minimal() +
  labs(title = "Charges by BMI Category and Smoker Status", x = "BMI Category", y = "Charges")
# Facet grid, Age vs Charges: BMI and Smoker Status
ggplot(insurance_data, aes(x = age, y = charges, color = bmi_category)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ smoker) +
  scale_color_manual(values = c("Underweight" = "blue", "Normal" = "green",
                                "Overweight" = "orange", "Obese" = "red")) +
  theme_minimal() +
  labs(title = "Age vs Charges Faceted by Smoker Status",
       x = "Age", y = "Charges")



# Set seed for reproducibility
set.seed(123)
# Split the data (80% training, 20% testing)
train_index <- createDataPartition(insurance_data$charges, p = 0.8, list = FALSE)
train_data <- insurance_data[train_index, ]
test_data <- insurance_data[-train_index, ]


# Regularized Regression(Ridge and Lasso)
x <- model.matrix(charges ~ ., data = train_data)[, -1]
y <- train_data$charges
ridge_model <- glmnet(x, y, alpha = 0)
plot(ridge_model, xvar = "lambda")
lasso_model <- glmnet(x, y, alpha = 1)
plot(lasso_model, xvar = "lambda")
# Multiple regression model
model <- lm(charges ~ age + bmi + children + smoker + region, data = train_data)
summary(model)
# Cross-validation
cv_results <- train(charges ~ ., data = train_data, method = "lm", trControl = trainControl(method = "cv"))
print(cv_results)



# Predict on the test set
predictions <- predict(model, newdata = test_data)
# Calculate RMSE
rmse <- sqrt(mean((predictions - test_data$charges)^2))
cat("RMSE:", rmse, "\n")
# Calculate R-squared
r_squared <- 1 - (sum((predictions - test_data$charges)^2) / sum((mean(train_data$charges) - test_data$charges)^2))
cat("R-squared:", r_squared, "\n")



# Residual plot
ggplot(data.frame(Residuals = residuals(model)), aes(x = Residuals)) +
  geom_histogram(binwidth = 500, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Residuals Distribution", x = "Residuals", y = "Frequency")
# Predicted vs. Actual
ggplot(data.frame(Predicted = predictions, Actual = test_data$charges), aes(x = Actual, y = Predicted)) +
  geom_point(color = "purple", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Predicted vs. Actual Charges", x = "Actual Charges", y = "Predicted Charges")
# Violin plot
ggplot(insurance_data, aes(x = smoker, y = charges, fill = smoker)) +
  geom_violin(alpha = 0.7) +
  scale_fill_manual(values = c("no" = "blue", "yes" = "red")) +
  theme_minimal() +
  labs(title = "Distribution of Charges by Smoker Status", x = "Smoker", y = "Charges")
# Random Forest
rf_model <- randomForest(charges ~ ., data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, newdata = test_data)


model_interaction <- lm(charges ~ age * smoker + bmi + children + region, data = train_data)
summary(model_interaction)


#------------------------------------------------------------------------------------------------------
# Deployment of the model
insurance_data$region <- as.factor(insurance_data$region)
model <- lm(charges ~ age + bmi + region, data = insurance_data)
predict(model, newdata = data.frame(age = insurance_data$age, bmi = insurance_data$bmi, region = insurance_data$region))


ui <- fluidPage(
  titlePanel("Medical Insurance Cost Predictor"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("age", "Age:", min = 18, max = 100, value = 30),
      selectInput("smoker", "Smoker:", choices = c("yes", "no")),
      numericInput("bmi", "BMI:", value = 25),
      numericInput("children", "Number of Children:", value = 0),
      selectInput("region", "Select Region:", choices = unique(insurance_data$region))
    ),
    mainPanel(
      textOutput("prediction")
    )
  )
)

server <- function(input, output) {
  model <- lm(charges ~ age + bmi + region + smoker + children, data = insurance_data)
  
  output$prediction <- renderText({
    validate(
      need(input$region != "", "Please select a region.")
    )
    
    new_data <- data.frame(
      age = input$age,
      smoker = input$smoker,
      bmi = input$bmi,
      children = input$children,
      region = input$region
    )
    
    predicted_value <- predict(model, newdata = new_data)
    paste("Predicted Charges: $", round(predicted_value, 2))
  })
}

shinyApp(ui, server)