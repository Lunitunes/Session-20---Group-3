import { z } from "zod"

export const uploadSchema = z.object({
  analysisName: z.string().min(1),
  file: z.file()
})