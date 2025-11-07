import { z } from "zod"

export const uploadSchema = z.object({
  analysisName: z
  .string()
  .min(1, "Name is Required")
  .max(15, "Can be no longer than 15 Characters")
  .regex(/^[a-zA-Z0-9 _-]+$/, "Only letters, numbers, spaces, hyphens, and underscores are allowed")
  .trim(),
  file: z
  .file()
  .mime(["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"], "File Must be a spreadsheet")
})